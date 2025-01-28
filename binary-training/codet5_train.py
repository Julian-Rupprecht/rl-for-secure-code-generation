from constants import BINARY_MODEL_OUTPUT_PATH_TO, BINARY_CRITIC_CONFIG_PATH, BINARY_CRITIC_MODEL_PATH, DEVIGN_TRAIN_DATASET_PATH, DEVIGN_TEST_DATASET_PATH, DEVIGN_VALID_DATASET_PATH
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, T5ForConditionalGeneration, T5Config, T5ForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import numpy as np
import jsonlines
import logging
from model import Model

logger = logging.getLogger(__name__)


def processDataset(path):
    with jsonlines.open(path, mode="r") as file:
        samples = []
        labels = []

        for line in file:
            samples.append(' '.join(line["func"].split())) 
            labels.append(line["target"])

        return samples, labels

class DevignDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        input = {key: val[idx] for key, val in self.samples.items()}
        input['labels'] = torch.tensor(self.labels[idx])
        return input


def train(model, device, num_epochs, train_dl, eval_dl, checkpoint_path): 
    model_save_path = checkpoint_path + '/model.bin'
    model.to(device)

    num_training_steps = num_epochs * len(train_dl)
    num_warmup_steps = num_training_steps * 0.05

    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    best_accuracy = 0.0
    model.zero_grad()

    logger.info("******* Start training *******")
    logger.info(f" Num iterations = {len(train_dl)}")
    logger.info(f" Num epochs = {num_epochs}")
    logger.info(f" Batch size = {train_dl.batch_size}")
    logger.info(f" Total training steps = {num_training_steps}")

    for epoch in range(1, num_epochs+1):
        training_iteration = 0
        training_loss = 0.0

        for batch in train_dl:  
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            model.train()
            loss, probs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            optimizer.zero_grad() # Sets gradients to zero so they dont accumulate after each pass
            loss.backward() 
            optimizer.step() # Adjusts weights:
            scheduler.step() # Adjusts learning rate for next iteration

            training_iteration += 1
            training_loss += loss.item()
  
        avg_loss = round(training_loss / training_iteration, 5)
        logger.info(f" Average loss of epoch {epoch} = {avg_loss}")

        results = evaluate(model, device, eval_dl)
        logger.info(f" Evaluation results of epoch {epoch}: loss = {results['eval_loss']}, accuracy = {results['eval_accuracy']}")
        
        # Save model checkpoint if the accuracy exceeds accuracy of previous checkpoints
        if results["eval_accuracy"] > best_accuracy:
            best_accuracy = results["eval_accuracy"]
            torch.save(model.state_dict(), model_save_path) #change path
            logger.info(f" Model checkpoint was saved to {model_save_path}")
              

def evaluate(model, device, eval_dl): 

    model.to(device)

    eval_loss = 0.0 
    eval_accuracy = 0.0

    logits = []
    labels = []

    logger.info("******* Start evaluation *******")
    logger.info(f" Num iterations = {len(eval_dl)}")
    logger.info(f" Batch size = {eval_dl.batch_size}")

    for batch in eval_dl:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['labels'].to(device)
        model.eval()

        with torch.no_grad():
            loss, probs = model(input_ids, attention_mask, label)
            eval_loss += loss.mean().item()
            logits.append(probs.cpu().numpy())
            labels.append(label.cpu().numpy())
      
    logits=np.concatenate(logits, 0)
    labels=np.concatenate(labels, 0)

    predictions = logits[:, 0] > 0.5
    eval_accuracy = np.mean(predictions == labels)
    eval_loss /= len(eval_dl)

    return { 
        "eval_loss": eval_loss,
        "eval_accuracy": eval_accuracy
    }


def test(checkpoint_path, model, device, test_dl):

    logger.info(f"******* Start test *******")
    logger.info(f" Num iterations = {len(test_dl)}")
    logger.info(f" Batch size = {test_dl.batch_size}")

    model_load_path = checkpoint_path + '/model.bin'
    predictions_path = checkpoint_path + '/predictions.txt'

    model.load_state_dict(torch.load(model_load_path, weights_only=True))
    model.to(device)

    logits = []

    for batch in test_dl:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            probs = model(input_ids, attention_mask)
            logits.append(probs.cpu().numpy())
       
    logits = np.concatenate(logits, 0)
    predictions = logits[:, 0] > 0.5

    with open(predictions_path, "w") as file:
        for pred in predictions: 
            file.writelines('1\n') if pred == True else file.writelines('0\n')

    
def main():
    
    config = T5Config.from_json_file(BINARY_CRITIC_CONFIG_PATH)

    model = T5ForSequenceClassification.from_pretrained(BINARY_CRITIC_MODEL_PATH, config=config)
    model = Model(model, config)

    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

    train_samples, train_labels = processDataset(DEVIGN_TRAIN_DATASET_PATH)
    test_samples, test_labels = processDataset(DEVIGN_TEST_DATASET_PATH)
    eval_samples, valid_labels = processDataset(DEVIGN_VALID_DATASET_PATH)

    train_tokens = tokenizer(train_samples, padding=True, truncation=True, max_length=510, return_tensors="pt", verbose=True)
    test_tokens = tokenizer(test_samples, padding=True, truncation=True, max_length=510, return_tensors="pt", verbose=True)
    eval_tokens = tokenizer(eval_samples, padding=True, truncation=True, max_length=510, return_tensors="pt", verbose=True)

    train_ds = DevignDataset(train_tokens, train_labels)
    test_ds = DevignDataset(test_tokens, test_labels)
    eval_ds = DevignDataset(eval_tokens, valid_labels)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)
    eval_dl = DataLoader(eval_ds, batch_size=16, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 5
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S')

    logger.info("******* Sample *******")
    logger.info(f" label: {train_labels[0]}")
    logger.info(f" input_tokens: {tokenizer.decode(train_tokens['input_ids'][0]).split()}")
    logger.info(f" input_ids: {train_tokens['input_ids'][0]}")

    checkpoint_path = f"{BINARY_MODEL_OUTPUT_PATH_TO}/run1"
    train(model, device, num_epochs, train_dl, eval_dl, checkpoint_path)
    
    result = evaluate(model, device, eval_dl)
    logger.info("******* Evaluation results *******")
    logger.info(f" evaluation loss = {result['eval_loss']}, evaluation accuracy = {result['eval_accuracy']}")
    
    test(checkpoint_path, model, device, test_dl)


if __name__ == "__main__":
    main() 