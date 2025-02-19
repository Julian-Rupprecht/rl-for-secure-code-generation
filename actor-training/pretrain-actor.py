import os
import argparse
import torch
from torch.utils.data import DataLoader
from helper import processDataset, SpocDataset, custom_collate_fn
from transformers import T5ForConditionalGeneration, RobertaTokenizerFast, get_linear_schedule_with_warmup
from torch.optim import AdamW
from model import Model
import logging
from codebleu import calc_codebleu


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

def train(model, train_dl, eval_dl, num_epochs, device, checkpoint_path, tokenizer):
    model_save_path = checkpoint_path + '/model.bin'

    num_training_steps = num_epochs * len(train_dl)
    num_warmup_steps = num_training_steps * 0.05

    optimizer = AdamW(model.parameters(), lr=3e-4, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    logger.info(" ******* Start Training *******")
    logger.info(f" Num iterations = {len(train_dl)}")
    logger.info(f" Num epochs = {num_epochs}")
    logger.info(f" Batch size = {train_dl.batch_size}")
    logger.info(f" Total training steps = {num_training_steps}")

    model.to(device)
    best_bleu = 0.0
    
    model.train() 
    for epoch in range(1, num_epochs+1):

        train_loss = 0.0
        i = 1
        for batch in train_dl:
            input_ids = batch['input_ids'].squeeze().to(device)
            attention_mask = batch['attention_mask'].squeeze().to(device)
            labels = batch['labels'].squeeze().to(device)

            loss, preds = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % 500 == 0:
                logger.info(f" Training step {i} of {len(train_dl)} in epoch {epoch}")
                logger.info(f" Allocated GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
                logger.info(f" Reserved GPU memory: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")

                
            train_loss += loss.item()
            i += 1

        
        train_loss /= len(train_dl)
        logger.info(f"Average training loss of epoch {epoch} = {train_loss}")

        # Evaluate after epoch
        results = evaluate(model, eval_dl, device, tokenizer)

        eval_bleu = results['eval_bleu']
        eval_loss = results['eval_loss']

        logger.info(f" Evaluation loss of epoch {epoch} = {eval_loss}")
        logger.info(f" Evaluation CodeBleu score of epoch {epoch} = {eval_bleu}")

        if eval_bleu > best_bleu:
            best_bleu = eval_bleu
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Model checkpoint was saved to {model_save_path}")


def evaluate(model, eval_dl, device, tokenizer):
    eval_loss = 0.0 
    eval_bleu = 0.0

    logger.info(" ******* Start Evaluation *******")
    logger.info(f" Num iterations = {len(eval_dl)}")
    logger.info(f" Batch size = {eval_dl.batch_size}")

    i = 1

    model.eval()
    for batch in eval_dl:
        input_ids = batch['input_ids'].squeeze().to(device)
        attention_mask = batch['attention_mask'].squeeze().to(device)
        labels = batch['labels'].squeeze().to(device)

        with torch.no_grad():
            loss, preds = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            preds = [tokenizer.decode(ids, skip_special_tokens=True) for ids in preds] 
            refs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]

        
            eval_loss += loss.item()
            score = calc_codebleu(references=refs, predictions=preds, lang="cpp")
            eval_bleu += score['codebleu']

            if i % 500 == 0:
                logger.info(f" Evaluation step {i} of {len(eval_dl)}")
                logger.info(f" Allocated GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
                logger.info(f" Reserved GPU memory: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
            i += 1


    eval_loss /= len(eval_dl)
    eval_bleu /= len(eval_dl)

    return {'eval_loss': eval_loss, 'eval_bleu': eval_bleu}


def test(model, test_dl, device, checkpoint_path, tokenizer):
    model_load_path = checkpoint_path + '/model.bin'
    predictions_path = checkpoint_path + '/predictions.txt'

    model.load_state_dict(torch.load(model_load_path, weights_only=True))

    test_loss = 0.0 
    test_bleu = 0.0

    logger.info(" ******* Start Testing *******")
    logger.info(f" Num iterations = {len(test_dl)}")
    logger.info(f" Batch size = {test_dl.batch_size}")

    i = 1
    predictions = []

    model.eval()
    for batch in test_dl:
        input_ids = batch['input_ids'].squeeze().to(device)
        attention_mask = batch['attention_mask'].squeeze().to(device)
        labels = batch['labels'].squeeze().to(device)

        model.eval()
        with torch.no_grad():
            loss, preds = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            preds = [tokenizer.decode(ids, skip_special_tokens=True) for ids in preds] 
            refs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]

            test_loss += loss.item()
            score = calc_codebleu(references=refs, predictions=preds, lang="cpp")
            test_bleu += score['codebleu']

            predictions.extend(preds)

            if i % 500 == 0:
                logger.info(f" Test step {i} of {len(test_dl)}")
                logger.info(f" Allocated GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
                logger.info(f" Reserved GPU memory: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
            i += 1

     
    test_loss /= len(test_dl)
    test_bleu /= len(test_dl)

    logger.info(f" Test loss = {test_loss}")
    logger.info(f" Test CodeBleu score = {test_bleu}")

    with open(predictions_path, "w") as file:
        for pred in predictions:
            file.writelines(f'{pred}\n')

    logger.info(f" Saved model predictions to {predictions_path}")

def main():

    parser.add_argument("--output_path", required=True, 
                        help="The path to save the model checkpoint and predictions")
    parser.add_argument("--num_epochs", default=5, type=int, 
                        help="The amount of epochs for training")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="The size of batches")
    parser.add_argument("--train_data_path", required=True, 
                        help="The path to the file containing the training data")
    parser.add_argument("--eval_data_path", required=True, 
                        help="The path to the file containing the evaluation data")
    parser.add_argument("--test_data_path", required=True, 
                        help="The path to the file containing the test data")

    args = parser.parse_args()

    checkpoint_path = args.output_path
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    train_data_path = args.train_data_path
    eval_data_path = args.eval_data_path
    test_data_path = args.test_data_path

    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S')

    tokenizer = RobertaTokenizerFast.from_pretrained("Salesforce/codet5-large")
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large")
    model = Model(model)

    train_text, train_code = processDataset(train_data_path)
    eval_text, eval_code = processDataset(eval_data_path)
    test_text, test_code = processDataset(test_data_path)

    train_ds = SpocDataset(train_text, train_code)
    eval_ds = SpocDataset(eval_text, eval_code)
    test_ds = SpocDataset(test_text, test_code)

    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True, collate_fn=custom_collate_fn)
    eval_dl = DataLoader(eval_ds, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True, collate_fn=custom_collate_fn)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False, collate_fn=custom_collate_fn)

    logger.info(" ******* Processed Dataset *******")
    logger.info(f" Sample Pseudo Code = {train_text[10]}")
    logger.info(f" Sample Code = {train_code[10]}")

    try: 
        os.mkdir(checkpoint_path)
        logger.info(f" Created folder at {checkpoint_path}")
    except Exception as e:
        logger.info(f" {e}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train model
    train(model, train_dl, eval_dl, num_epochs, device, checkpoint_path, tokenizer)
    # evaluate model
    # test model
    test(model, test_dl, device, checkpoint_path, tokenizer)    
    return 0

if __name__ == "__main__":
    main()
