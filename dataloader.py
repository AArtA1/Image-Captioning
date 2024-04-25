import os
import json

import torch
from torchvision.transforms import v2
from PIL import Image
from datasets import load_dataset, load_from_disk
from torch.nn.utils.rnn import pad_sequence


class Flickr8Dataset(torch.utils.data.Dataset):
    """
        Custom Dataset for the Flickr8k 
    """
    def __init__(self, dataset_path, tokenizer, transform, max_seq_len):
        self._dataset = load_from_disk(dataset_path)

        self._max_seq_len = max_seq_len
        self._transform = transform
        self._tokenizer = tokenizer

        # def process(example):
        #     example['image'] = [transform(val) for val in example['image']]
        #     #example['text'] = tokenizer.encode(example['text'])
        #     return example

        #self._dataset = self._dataset.map(process, batched = True, batch_size = 64)

    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, index):
        # Extract the caption data
        sample = self._dataset[index]

        image_tensor = self._transform(sample['image'])
        text = sample['text']

        tokenized_input = self._tokenizer.encode(text)

        tokenized_output = tokenized_input[1:]

        tokenized_input = tokenized_input[:-1]

        tokenized_input = torch.tensor(tokenized_input + [0] * (self._max_seq_len - len(tokenized_input)))

        tokenized_output = torch.tensor(tokenized_output + [0] * (self._max_seq_len - len(tokenized_output)))

        return image_tensor, tokenized_input, tokenized_output