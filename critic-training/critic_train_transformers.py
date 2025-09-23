from transformers import AutoModelForSequenceClassification , T5Config, AutoConfig, RobertaTokenizer, Trainer, TrainingArguments
import torch
import torch.nn as nn
import jsonlines
import logging 
import argparse
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S')

logger = logging.getLogger(__name__)


def process_reposvul_identity_pairs(pathToDataset):
    samples = []

    with jsonlines.open(pathToDataset, mode="r") as file:
        for line in file:
            index = int(line["index"].strip())
            cve_id = line["cve_id"].strip()
            cve_language = line["cve_language"].strip()
            cve_description = line["cve_description"].strip()
            function_before = line["function_before"].strip()
            function_after = line["function_after"].strip()
            target = int(line["target"])

            
            samples.append({
                "input": function_before,
                "label": target,
            })

    return samples



class ReposVulDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, tokenizer):
        self.samples = process_reposvul_identity_pairs(filepath)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text = sample["input"]
        label = sample["label"]

        input_tokens = self.tokenizer(input_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

        item = {
            "input_ids": input_tokens["input_ids"].squeeze(0),
            "attention_mask": input_tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor(label),
        }

        return item


def processDataset(path):
    with jsonlines.open(path, mode="r") as file:
        samples = []
        labels = []

        for line in file:
            samples.append(' '.join(line["func"].split())) 
            labels.append(line["target"])

        return samples, labels


class DevignDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx): 
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


def custom_loss_func(outputs, labels, num_items_in_batch):
    criterion = nn.BCEWithLogitsLoss()
    labels = labels.unsqueeze(1).float()
    loss = criterion(outputs['logits'], labels)

    return loss


def custom_compute_metric_func(eval_prediction):
    logits = eval_prediction.predictions[0]
    labels = eval_prediction.label_ids

    logits = torch.tensor(logits)

    preds = torch.sigmoid(logits)[:, 0] > 0.5
    preds = preds.cpu().numpy()
    
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)


    return { "accuracy": accuracy, "f1": f1 }


parser = argparse.ArgumentParser()

def main(): 
    parser.add_argument("--output_path", required=True, 
                        help="The path to save the model checkpoint and predictions")
    parser.add_argument("--config_path", required=True,
                        help="The path to the model config file") 
    parser.add_argument("--model_path", required=True,
                        help="The path to the model checkpoint")
    parser.add_argument("--train_data_path", required=True, 
                        help="The path to the file containing the training data")
    parser.add_argument("--eval_data_path", required=True, 
                        help="The path to the file containing the evaluation data")
    parser.add_argument("--test_data_path", required=True, 
                        help="The path to the file containing the test data")
    parser.add_argument("--seed", default=42, type=int,
                        help="The seed for deterministic training")
    parser.add_argument("--dataset", default="Devign", type=str,
                        help="The seed for deterministic training")
    args = parser.parse_args()


    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

    train_samples, train_labels = processDataset(args.train_data_path)
    eval_samples, eval_labels = processDataset(args.eval_data_path)
    test_samples, test_labels = processDataset(args.test_data_path)

    train_tokens = tokenizer(train_samples, padding=True, truncation=True, max_length=510, return_tensors="pt", verbose=True)
    test_tokens = tokenizer(test_samples, padding=True, truncation=True, max_length=510, return_tensors="pt", verbose=True)
    eval_tokens = tokenizer(eval_samples, padding=True, truncation=True, max_length=510, return_tensors="pt", verbose=True)

    train_ds = DevignDataset(train_tokens, train_labels)
    eval_ds = DevignDataset(eval_tokens, eval_labels)
    test_ds = DevignDataset(test_tokens, test_labels)

    #train_ds = ReposVulDataset(args.train_data_path, tokenizer)
    #eval_ds = ReposVulDataset(args.eval_data_path, tokenizer)
    #test_ds = ReposVulDataset(args.test_data_path, tokenizer)
    
    config = T5Config.from_json_file(args.config_path)
    #config = AutoConfig.from_pretrained(args.config_path, num_labels=1, problem_type="single_label_classification")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, config=config)
    
    output_dir = f"{args.output_path}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        do_predict=True,
        eval_strategy="epoch",
        prediction_loss_only=False,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5, 
        save_strategy="epoch",
        learning_rate=5e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_loss_func=custom_loss_func,
        compute_metrics=custom_compute_metric_func
    )

    logger.info(f" ******* Start training {args.dataset} *******")
    trainer.train()

    logger.info(f" ******* Start evaluation *******")
    trainer.evaluate()

    logger.info(f" ******* Start testing *******")
    predictions = trainer.predict(test_ds)
    logger.info(f" Results of test")
    logger.info(f" {predictions.metrics}")

    logger.info(f" ******* Saving model *******")
    model.save_pretrained(output_dir)
    logger.info(f" Model was saved to {output_dir}")


if __name__ == "__main__":
    main()