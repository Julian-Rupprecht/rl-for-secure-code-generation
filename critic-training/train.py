from transformers import T5ForConditionalGeneration, AutoTokenizer,Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.utils.data
import jsonlines
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def processDataset(path):
    samples, labels = [], []
    with jsonlines.open(path, mode="r") as file:
        for line in file:
            samples.append(" ".join(line["func"].split()))
            labels.append(int(line["target"]))
    return samples, labels


class DevignSeq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, samples, labels, max_input_len=512, max_output_len=10):
        self.tokenizer = tokenizer
        self.samples = samples
        self.labels = labels
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_text = f"<func> {self.samples[idx]} </func> <cls>"
        target_text = "true" if self.labels[idx] == 1 else "false"

        input_enc = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_input_len,
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_output_len,
            return_tensors="pt",
        )

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": target_enc["input_ids"].squeeze(),
        }


import numpy as np

def compute_metrics_seq2seq(eval_preds):
    preds, labels = eval_preds


    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.array(preds)
    labels = np.array(labels)
  
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    preds_binary = [1 if pred.strip().lower() == "true" else 0 for pred in decoded_preds]
    labels_binary = [1 if label.strip().lower() == "true" else 0 for label in decoded_labels]

    accuracy = accuracy_score(labels_binary, preds_binary)
    f1 = f1_score(labels_binary, preds_binary)

    return {"accuracy": accuracy, "f1": f1}




parser = argparse.ArgumentParser()


def main():
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--eval_data_path", required=True)
    parser.add_argument("--test_data_path", required=True)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset", default="Devign", type=str)
    args = parser.parse_args()

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")

    train_samples, train_labels = processDataset(args.train_data_path)
    eval_samples, eval_labels = processDataset(args.eval_data_path)
    test_samples, test_labels = processDataset(args.test_data_path)

    train_ds = DevignSeq2SeqDataset(tokenizer, train_samples, train_labels)
    eval_ds = DevignSeq2SeqDataset(tokenizer, eval_samples, eval_labels)
    test_ds = DevignSeq2SeqDataset(tokenizer, test_samples, test_labels)

    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")

    training_args = TrainingArguments(
        output_dir=args.output_path,
        do_train=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics_seq2seq,
    )

    logger.info(f"******* Start training {args.dataset} *******")
    trainer.train()

    logger.info(f"******* Start evaluation *******")
    trainer.evaluate()

    logger.info(f"******* Start testing *******")
    predictions = trainer.predict(test_ds)
    logger.info(f"Test Results: {predictions.metrics}")

    logger.info(f"******* Saving model *******")
    model.save_pretrained(args.output_path)
    logger.info(f"Model saved to {args.output_path}")


if __name__ == "__main__":
    main()
