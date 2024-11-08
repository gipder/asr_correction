import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from peft import LoraConfig, get_peft_model

## 1. 모델과 토크나이저 로드
#model_name = "huggingface/llama-3"  # 실제 LLaMA-3 분류 모델로 대체해야 함
#tokenizer = AutoTokenizer.from_pretrained(model_name)
# 1. 모델과 토크나이저 준비
model_name = "meta-llama/Meta-Llama-3-8B"  # 모델 경로 (로컬이거나 Hugging Face hub에서 불러오기)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                           load_in_8bit=True,
                                                           device_map="auto",
                                                           num_labels=2)  # 2개의 레이블 (긍정, 부정)
model.config.pad_token_id = model.config.eos_token_id

# LoRA adapter
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)
# 2. IMDb 데이터셋 로드
dataset = load_dataset("imdb")

# 데이터 샘플링 (학습 시간을 줄이기 위해 작은 데이터셋으로 작업)
small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
small_test_dataset = dataset["test"].shuffle(seed=42).select(range(200))

# 3. 데이터 전처리 함수 정의
def preprocess_data(batch):
    # 입력 텍스트 토크나이즈
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    inputs["labels"] = batch["label"]
    #inputs["labels"] = torch.tensor(batch["label"], dtype=torch.long)  # 레이블 추가
    #print("inputs: ", inputs)
    return inputs

# 데이터셋 전처리 적용
train_dataset = small_train_dataset.map(preprocess_data, batched=True)
test_dataset = small_test_dataset.map(preprocess_data, batched=True)

# torch 텐서 형식으로 변환
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 4. 데이터 콜레이터 정의
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4. 훈련 파라미터 설정
training_args = TrainingArguments(
    output_dir="./llama3-finetuned-imdb",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
)

# 5. Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

print("Before fine-tuning")

# 8. 모델 평가
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 1. 테스트 데이터셋에 대한 예측 수행
#print("test_dataset: ", test_dataset)
predictions = trainer.predict(test_dataset)
#print("predictions.predictions: ", predictions.predictions)
pred_labels = predictions.predictions[1].argmax(axis=1)  # 예측된 레이블
#print(f"{pred_labels=}")
#print(f"{test_dataset['labels']=}")
#true_labels = predictions.label_ids  # 실제 레이블
true_labels = test_dataset['labels']

# 2. 정확도 계산
accuracy = accuracy_score(true_labels, pred_labels)
print(f"Accuracy: {accuracy:.4f}")

# 3. 분류 보고서 출력
report = classification_report(true_labels, pred_labels, target_names=["Negative", "Positive"])
print("Classification Report:\n", report)
print("*" * 100)

# 6. Fine-tuning 실행
trainer.train()

# 7. 모델 평가
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 1. 테스트 데이터셋에 대한 예측 수행
predictions = trainer.predict(test_dataset)
#print("predictions.predictions: ", predictions.predictions)
pred_labels = predictions.predictions[1].argmax(axis=1)  # 예측된 레이블
#true_labels = predictions.label_ids  # 실제 레이블
true_labels = test_dataset['labels']

# 2. 정확도 계산
accuracy = accuracy_score(true_labels, pred_labels)
print(f"Accuracy: {accuracy:.4f}")

# 3. 분류 보고서 출력
report = classification_report(true_labels, pred_labels, target_names=["Negative", "Positive"])
print("Classification Report:\n", report)

