import torch 
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast


tokenizer = RobertaTokenizerFast.from_pretrained("Salesforce/codet5-large")

def processDataset(pathToDataset):
    text = []
    code = []

    with open(pathToDataset, "r") as file:
        for line in file: 
            data = line.split("\t")
            text.append(data[0])
            code.append(data[1])
        
    return text[1:], code[1:] 

class SpocDataset(Dataset):
    def __init__(self, text, code):
        self.text = text
        self.code = code
    
    def __len__(self):
        return len(self.code)
    
    def __getitem__(self, idx):
        #text_tokens = tokenizer(self.text[idx], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        #code_tokens = tokenizer(self.code[idx], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        
        text_tokens = tokenizer(self.text[idx])
        code_tokens = tokenizer(self.code[idx])

        input = {key: val for key, val in text_tokens.items()}
        input['labels'] = code_tokens['input_ids']

        return input

def custom_collate_fn(batch):
    max_length_text = max(len(item['input_ids']) for item in batch)
    max_length_code = max(len(item['labels']) for item in batch)
    
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for item in batch:
        pad_length_text = max_length_text - len(item['input_ids'])
        pad_length_code = max_length_code - len(item['labels'])

        padded_input_ids = item['input_ids'] + [0] * pad_length_text
        padded_attention_mask = [1] * len(item['input_ids']) + [0] * pad_length_text
        padded_label = item['labels'] + [0] * pad_length_code

        batch_input_ids.append(padded_input_ids)
        batch_attention_mask.append(padded_attention_mask)
        batch_labels.append(padded_label)

    return {
        "input_ids": torch.tensor(batch_input_ids),
        "attention_mask": torch.tensor(batch_attention_mask),
        "labels": torch.tensor(batch_labels)
    }


