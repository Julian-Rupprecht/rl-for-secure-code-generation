import pickle
import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast
from torch.utils.data import Dataset
import numpy as np 
import csv
import jsonlines

def process_reposvul_identity_pairs(pathToDataset):
    samples = []

    with jsonlines.open(pathToDataset, mode="r") as file:
        for line in file:
            index = int(line["index"].strip())
            cwe_id = line["cwe_id"].strip()
            cve_language = line["cve_language"].strip()
            cve_description = line["cve_description"].strip()
            function_before = line["function_before"].strip()
            function_after = line["function_after"].strip()
            target = int(line["target"])

            tag = f"<vuln> <cwe-id>{cwe_id}" if target == 1 else "<safe>"
            input_text = f"{tag} <code>\n{function_before}"

            samples.append({
                "input": input_text,
                "target": function_before,
                "label": target,
                "index": index,
                "function_after": function_after,
                "cve_language": cve_language,
                "cve_description": cve_description
            })

    return samples

class ReposVulDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        self.samples = process_reposvul_identity_pairs(filepath)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text = sample["input"]
        target_text = sample["target"]

        input_tokens = self.tokenizer(input_text, padding="max_length", truncation=True, max_length=512)
        label_tokens = self.tokenizer(target_text, padding="max_length", truncation=True, max_length=512)

        item = {
            "input_ids": torch.tensor(input_tokens["input_ids"]),
            "attention_mask": torch.tensor(input_tokens["attention_mask"]),
            "labels": torch.tensor(label_tokens["input_ids"]),
            "function_after": sample["function_after"],
            "index": sample["index"],
            "target": sample["label"],
            "cve_language": sample["cve_language"],
        }

        return item

def reposvul_collate_fn(batch):
    collated = {}

    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]
    
    return collated

def processSPOCDatasetFunctionWise(pathToDataset):
    text = []
    code = []
    sub_ids = []

    func = ''
    gold_func = ''

    with open(pathToDataset, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader)

        for data in reader:
            if len(data) < 2:
                continue
                
            pseudo, gold_code, sub_id, line, indent = data[0].strip(), data[1].strip(), int(data[4].strip()), int(data[5].strip()), int(data[6].strip())

            if line == 0 and func != '':
                text.append(func)
                code.append(gold_func)
                
                func = ''
                gold_func = ''
            
            
            if pseudo == '':
                func += '\t' * indent + gold_code + '\n'
            else:
                func += '\t' * indent + pseudo + '\n'
            
            gold_func += '\t' * indent + gold_code + '\n'

            if line == 0:
                sub_ids.append(sub_id)

    return text[:10], code[:10], sub_ids[:10]

def processSPOCDataset(pathToDataset):
    text = []
    code = []
    lines = []
    indents = []

    with open(pathToDataset, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")  
        next(reader)

        for data in reader:
            if len(data) < 2: 
                continue  

            pseudo, gold_code, line, indent = data[0].strip(), data[1].strip(), data[5].strip(), data[6].strip()

            if pseudo == "":
                text.append(gold_code)
            else:
                text.append(pseudo)
            
            code.append(gold_code)
            lines.append(line)
            indents.append(indent)

    return text, code, lines, indents

def custom_collate_fn(batch):
    max_length_text = max(len(input_ids) for input_ids in batch['input_ids'])
    
    batch_input_ids = []

    for input_ids in batch['input_ids']:
        pad_length_text = max_length_text - len(input_ids)
        padded_input_ids = input_ids + [0] * pad_length_text
        batch_input_ids.append(padded_input_ids)

    return {
        'input_ids': torch.tensor(batch_input_ids),
        'text': batch['text'],
        'code': batch['code'],
        'lines': batch['lines'],
        'indents': batch['indents']
    }

def custom_collate_fn2(batch):
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

tokenizer = RobertaTokenizerFast.from_pretrained("Salesforce/codet5-large")

class SpocDataset(Dataset):
    def __init__(self, filepath):
        self.text, self.code, self.lines, self.indents = processSPOCDataset(filepath)

    def __len__(self):
        return len(self.indents)
    
    def __getitem__(self, idx):
        tokens = tokenizer(self.text[idx], return_attention_mask=False)
        tokens.update({
            "text": self.text[idx],
            "code": self.code[idx],
            "lines": int(self.lines[idx]),
            "indents": int(self.indents[idx])
        })
        return tokens

class SpocDatasetFunctionWise(Dataset):
    def __init__(self, filepath, tokenizer):
        self.text, self.code, self.sub_ids = processSPOCDatasetFunctionWise(filepath)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.code)
    
    def __getitem__(self, idx):
        text_tokens = self.tokenizer(self.text[idx], padding="max_length", return_tensors="pt", max_length=512, truncation=True, return_attention_mask=False)
        code_tokens = self.tokenizer(self.code[idx], padding="max_length", return_tensors="pt", max_length=512, truncation=True, return_attention_mask=False)
       
        input = {key: torch.tensor(val) for key, val in text_tokens.items()}
        input['labels'] = torch.tensor(code_tokens['input_ids'])
        input['sub_ids'] = self.sub_ids[idx]

        return input

class ActorModel(nn.Module):
    def __init__(self, model):
        super(ActorModel, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        #loss = output.loss
        #preds = torch.argmax(output.logits, dim=-1)
        #return loss, preds

        return output
    
    def generate(self, input_ids, max_length=512, **generation_kwargs):
        return self.model.generate(input_ids, max_length=max_length, **generation_kwargs)

class CriticModel(nn.Module):
    def __init__(self, model, config):
        super(CriticModel, self).__init__()
        self.model = model
        self.config = config
        #self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, return_hidden_states=False): 
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=return_hidden_states)
  
        #outputs = self.dropout(outputs)

        probs = torch.sigmoid(outputs.logits)

        if return_hidden_states:
            return probs, outputs.encoder_hidden_states[-1]

        return probs
    
