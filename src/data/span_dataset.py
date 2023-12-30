import logging

import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpanDataset(Dataset):
    def __init__(self, data: Dataset, tokenizer, batch_size, max_seq_length, shuffle=False):
        """
        :type data: datasets.arrow_dataset.Dataset
        :type tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast
        """
        self.data = data
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.text_field_name = "text"  # TODO this should be made customizable

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def tokenize(self, batch):
        return self.tokenizer(batch[self.text_field_name], padding='max_length', truncation=True)

    def encode(self):
        # batch_size=None will tokenize the entire dataset at once, ensuring we pad each example to be the same length
        self.data = self.data.map(self.tokenize, batched=True, batch_size=None)
        self.data.set_format('torch')

    def data_loader(self):
        return torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=self.shuffle)
