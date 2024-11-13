import os
import sys
import time
import torch
#adding import about List
from typing import List, Dict, Tuple, Any
from typing import Optional
import logging

def get_batch(
    data: List[Dict],
    batch_size: int,
    #longest_seq_length: int,
    #longest_seq_ix: Optional[int] = None,
    #max_input_length: Optional[int] = 1024,
    device: Optional[torch.device] = torch.device("cuda"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    data: A list that contains dictionaries of pytorch data
    batch_size: The size of the batch
    #longest_seq_length: The longest sequence length in the batch
    #longest_seq_ix: The index of the longest sequence in the batch
    #max_input_length: The maximum input length for the model, e.g. 1024
    device: The device to use, e.g. cuda or cpu
    """
    # Initialize tensors
    max_input_length = get_max_seq_length(data)[1]
    if batch_size > len(data):
        batch_size = len(data)

    input_ids =[data[i]["input_ids"][:max_input_length].type(torch.int64) for i in range(batch_size)]
    labels = [data[i]["labels"][:max_input_length].type(torch.int64) for i in range(batch_size)]

    def pad_right(x, pad_id, max_len=max_input_length) -> torch.Tensor:
        n = max_len - len(x)
        return torch.cat([x, torch.tensor([pad_id]*n)])

    x = torch.stack([pad_right(i, 0) for i in input_ids]).to(device)
    y = torch.stack([pad_right(i, -1) for i in labels]).to(device)
    logging.debug(f"x shape: {x.shape}, y shape: {y.shape}")
    logging.debug(f"{x[-1]=}")
    logging.debug(f"{y[-1]=}")
    return x, y

def get_max_seq_length(data: List[Dict]) -> Tuple[int, int, int]:
    # find out the minimum max_seq_length
    override_max_seq_length = None
    lengths = [len(d['input_ids']) for d in data]
    max_seq_length = max(lengths)
    longest_seq_ix = lengths.index(max_seq_length)
    # return Tuple
    return (
        override_max_seq_length if isinstance(override_max_seq_length, int) else max_seq_length,
        max_seq_length,
        longest_seq_ix,
    )

import unittest

class TestMyUtils(unittest.TestCase):
    # initialize inner variables and other things
    DATA_PATH = "./data/clean.pt"
    BATCH_SIZE = 4
    LONGEST_SEQ_LENGTH = 168
    LONGEST_SEQ_IX = 23
    MAX_INPUT_LENGTH = 1024

    def setUp(self):
        self.data = torch.load(self.DATA_PATH, weights_only=False)
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    def test_get_max_seq_length(self):
        results = get_max_seq_length(self.data)
        override_max_seq_length = results[0]
        max_seq_length = results[1]
        longest_seq_ix = results[2]

        self.assertNotEqual(override_max_seq_length, None)
        self.assertEqual(self.LONGEST_SEQ_LENGTH, override_max_seq_length)
        self.assertEqual(self.LONGEST_SEQ_LENGTH, max_seq_length)
        self.assertEqual(self.LONGEST_SEQ_IX, longest_seq_ix)

    def test_get_batch(self):
        for i in range(0, len(self.data), self.BATCH_SIZE):
            data = self.data[i:i+self.BATCH_SIZE]
            x, y = get_batch(
                data,
                self.BATCH_SIZE,
            )

if __name__ == "__main__":

    print("Running Unit Tests\n" + "="*20)
    unittest.TextTestRunner(verbosity=1).run(
        unittest.TestLoader().loadTestsFromTestCase(TestMyUtils)
    )

