import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, labels=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        if self.labels:
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.long)
            return { 'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}
        return { 'input_ids': input_ids, 'attention_mask': attention_mask}