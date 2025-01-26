from constants import BINARY_CRITIC_MODEL_PATH, BINARY_CRITIC_CONFIG_PATH, DEVIGN_TRAIN_DATASET_PATH, DEVIGN_TEST_DATASET_PATH, DEVIGN_VALID_DATASET_PATH, BINARY_MODEL_OUTPUT_PATH_TR
from transformers import T5ForSequenceClassification, T5Config, RobertaTokenizer, Trainer, TrainingArguments
import torch
import torch.nn as nn
import numpy as np
import jsonlines
import logging 

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S')

logger = logging.getLogger(__name__)


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
        item['label'] = torch.tensor(self.labels[idx])
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
    accuracy = np.mean(preds==labels)

    return { "accuracy": accuracy }


def main(): 
    train_samples, train_labels = processDataset(DEVIGN_TRAIN_DATASET_PATH)
    eval_samples, eval_labels = processDataset(DEVIGN_VALID_DATASET_PATH)
    test_samples, test_labels = processDataset(DEVIGN_TEST_DATASET_PATH)

    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

    train_tokens = tokenizer(train_samples, padding=True, truncation=True, max_length=510, return_tensors="pt", verbose=True)
    test_tokens = tokenizer(test_samples, padding=True, truncation=True, max_length=510, return_tensors="pt", verbose=True)
    eval_tokens = tokenizer(eval_samples, padding=True, truncation=True, max_length=510, return_tensors="pt", verbose=True)

    train_ds = DevignDataset(train_tokens, train_labels)
    eval_ds = DevignDataset(eval_tokens, eval_labels)
    test_ds = DevignDataset(test_tokens, test_labels)

    config = T5Config.from_json_file(BINARY_CRITIC_CONFIG_PATH)
    model = T5ForSequenceClassification.from_pretrained(BINARY_CRITIC_MODEL_PATH, config=config)
    
    output_dir = f"{BINARY_MODEL_OUTPUT_PATH_TR}/run1"
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        do_predict=True,
        eval_strategy="epoch",
        prediction_loss_only=False,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        num_train_epochs=5, 
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_loss_func=custom_loss_func,
        compute_metrics=custom_compute_metric_func
    )

    logger.info(f" ******* Start training *******")
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