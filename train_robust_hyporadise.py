import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, LlamaForCausalLM, AdamW
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from peft import LoraConfig, get_peft_model, TaskType
import my_utils

class MaskedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        valid_masks = (labels > -1)
        # 모델 출력
        outputs = model(**inputs)
        logits = outputs.logits
        print(logits)
        # Cross Entropy 손실 계산
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')  # 요소별 손실 계산
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # 마스크 적용
        if labels_mask is not None:
            mask = labels_mask.view(-1).float()
            loss = loss * mask  # 마스킹된 부분은 0으로 만들어 손실에서 제외
            loss = loss.sum() / mask.sum()  # 평균 손실 계산 (마스크된 부분 제외)

        return (loss, outputs) if return_outputs else loss

# 1. 모델과 토크나이저 준비
# 일단은 Llama-2를 사용 왜냐하면 학습 DB에 Tokenizer가 Llama-2로 되어 있기 때문
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(model_name,
                                         torch_dtype=torch.bfloat16,
                                         device_map="auto")

model.config.pad_token_id = model.config.eos_token_id
# LoRA adapter
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)

# 2. 데이터셋 로드
test_data = torch.load("data/clean.pt")
data = torch.load("/home/sskim/work/Robust-HyPoradise/train_noizeus.pt")
dataset = Dataset.from_list(data)
test_dataset = Dataset.from_list(test_data)

# 데이터 샘플링 (학습 시간을 줄이기 위해 작은 데이터셋으로 작업)
small_train_dataset = dataset.shuffle(seed=42)
small_test_dataset = test_dataset.shuffle(seed=42)

# 3. 데이터 전처리 함수 정의
def preprocess_data(batch):
    # Attention mask 생성
    attention_mask = [[1] * len(input_ids) for input_ids in batch["input_ids"]]
    labels = [[-100 if label == -1 else label for label in labels] for labels in batch["labels"]]
    # Labels mask 생성
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": attention_mask,
        "labels": labels,
    }

# 데이터셋 전처리 적용
train_dataset = small_train_dataset.map(preprocess_data, batched=True)
test_dataset = small_test_dataset.map(preprocess_data, batched=True)

# torch 텐서 형식으로 변환
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 데이터 콜레이터 정의
def collate_fn(batch, max_length=256):
    # get the max length of the batch
    max_input_length = max(len(item["input_ids"]) for item in batch)
    max_label_length = max(len(item["labels"]) for item in batch)
    if max_input_length > max_length:
        max_input_length = max_length
    if max_label_length > max_length:
        max_label_length = max_length

    # padding with tensor in torch
    input_ids_type = batch[0]["input_ids"].dtype
    attention_mask_type = batch[0]["attention_mask"].dtype
    labels_type = batch[0]["labels"].dtype

    padded_input_ids = torch.zeros((len(batch), max_input_length), dtype=input_ids_type)
    # fill with -100 for labels
    padded_labels = torch.full((len(batch), max_label_length), -100, dtype=labels_type)
    padded_attention_mask = torch.zeros((len(batch), max_input_length), dtype=attention_mask_type)

    for idx in range(len(batch)):
        input_length = min(len(batch[idx]["input_ids"]), max_input_length)
        label_length = min(len(batch[idx]["labels"]), max_label_length)

        padded_input_ids[idx, :input_length] = batch[idx]["input_ids"][:input_length]
        padded_labels[idx, :label_length] = batch[idx]["labels"][:label_length]
        padded_attention_mask[idx, :input_length] = batch[idx]["attention_mask"][:input_length]

    return {
        "input_ids": padded_input_ids.clone().detach(),
        "labels": padded_labels.clone().detach(),
        "attention_mask": padded_attention_mask.clone().detach()
    }

# Testing the function collate_fn

# 4. 훈련 파라미터 설정
training_args = TrainingArguments(
    output_dir="./llama2-lora-finetuned",
    per_device_train_batch_size=5,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=5e-4,
    save_steps=50,
)

# 5. Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

# 시작전에 Testing 해보기
for item in test_dataset:
    prompt_mask = (item['labels'] == -100)
    # get prompt with prompt mask
    prompt = item['input_ids'] * prompt_mask
    attention_mask = item['attention_mask'] * prompt_mask
    inputs = dict()
    inputs['input_ids'] = prompt.unsqueeze(0).to(model.device)
    inputs['attention_mask'] = attention_mask.unsqueeze(0).to(model.device)

    output = model.generate(**inputs, max_length=258)
    # shift left by the length of the prompt
    # the length of the prompt can be calculated from the number of -100 in the labels
    # get the length of the prompt
    prompt_length = prompt_mask.sum()
    input_text = tokenizer.decode(inputs['input_ids'][0, :prompt_length], skip_special_tokens=True)
    output = output[0, prompt_length:]
    print("*** Input text: \n", input_text)
    print("*** Generated output: \n", tokenizer.decode(output, skip_special_tokens=True))
    break

# 6. Fine-tuning 실행
trainer.train()

print("After training")
print("=" * 10)
for item in test_dataset:
    prompt_mask = (item['labels'] == -100)
    # get prompt with prompt mask
    prompt = item['input_ids'] * prompt_mask
    attention_mask = item['attention_mask'] * prompt_mask
    inputs = dict()
    inputs['input_ids'] = prompt.unsqueeze(0).to(model.device)
    inputs['attention_mask'] = attention_mask.unsqueeze(0).to(model.device)

    output = model.generate(**inputs, max_length=258)
    # shift left by the length of the prompt
    # the length of the prompt can be calculated from the number of -100 in the labels
    # get the length of the prompt
    prompt_length = prompt_mask.sum()
    input_text = tokenizer.decode(inputs['input_ids'][0, :prompt_length], skip_special_tokens=True)
    output = output[0, prompt_length:]
    print("*** Input text: \n", input_text)
    print("*** Generated output: ", tokenizer.decode(output, skip_special_tokens=True))
    break
