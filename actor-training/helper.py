import torch 
from torch.utils.data import Dataset
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
                "target": function_after,
                "label": target,
                "index": index
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

        input_tokens = self.tokenizer(input_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        label_tokens = self.tokenizer(target_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

        item = {
            "input_ids": input_tokens["input_ids"].squeeze(0),
            "attention_mask": input_tokens["attention_mask"].squeeze(0),
            "labels": label_tokens["input_ids"].squeeze(0),
            "index": sample["index"],
            "target": sample["label"],
        }

        return item


def processDataset(pathToDataset):
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

            pseudo, gold_code, line, indent = data[0].strip(), data[1].strip(), int(data[5].strip()), int(data[6].strip())

            if pseudo == "":
                text.append(gold_code)
            else:
                text.append(pseudo)
            
            code.append(gold_code)
            lines.append(line)
            indents.append(indent)

    return text, code, lines, indents
  
def processDatasetFunctionWise(pathToDataset):
    text = []
    code = []

    func = ''
    gold_func = ''

    with open(pathToDataset, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader)

        for data in reader:
            if len(data) < 2:
                continue
                
            pseudo, gold_code, line, indent = data[0].strip(), data[1].strip(), int(data[5].strip()), int(data[6].strip())

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

    return text, code
  
class SpocDatasetFunctionWise(Dataset):
    def __init__(self, filepath, tokenizer):
        self.text, self.code = processDatasetFunctionWise(filepath)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.code)
    
    def __getitem__(self, idx):
        text_tokens = self.tokenizer(self.text[idx], padding="max_length", truncation=True, max_length=512)
        code_tokens = self.tokenizer(self.code[idx], padding="max_length", truncation=True, max_length=512)
       
        input = {key: torch.tensor(val) for key, val in text_tokens.items()}
        input['labels'] = torch.tensor(code_tokens['input_ids'])

        return input

class SpocDatasetLineWise(Dataset):
    def __init__(self, filepath, tokenizer):
        self.text, self.code, self.lines, self.indents = processDataset(filepath)
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.code)
    
    def __getitem__(self, idx):
        text_tokens = self.tokenizer(self.text[idx])
        code_tokens = self.tokenizer(self.code[idx])

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