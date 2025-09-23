import argparse
import logging
import torch
import os
import numpy as np 
from generation_dataset import GenerationDataset
from helper import ActorModel, CriticModel, ReposVulDataset, reposvul_collate_fn
from transformers import T5ForConditionalGeneration, T5ForSequenceClassification, RobertaTokenizerFast, get_linear_schedule_with_warmup, AutoTokenizer
from torch.utils.data import DataLoader
from codebleu import calc_codebleu


parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)


def compute_loss2(logits, labels, rewards):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1) 
    selected_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1))  # [B, T, 1]
    selected_log_probs = torch.clamp(selected_log_probs, min=-10.0, max=0.0)

    mask = (labels != 0).unsqueeze(-1)  # [B, T, 1]


    weighted_log_probs = selected_log_probs * rewards 


    masked_log_probs = weighted_log_probs[mask]  

    rl_loss = -masked_log_probs.mean()

    logger.info(f"RL Loss: {rl_loss.item():.4f}")
    logger.info(f"Log prob stats: min={selected_log_probs.min().item():.4f}, max={selected_log_probs.max().item():.4f}")
    logger.info(f"Reward stats: mean={rewards.mean().item():.4f}, std={rewards.std().item():.4f}, min={rewards.min().item():.4f}, max={rewards.max().item():.4f}")

    return rl_loss



def compute_loss(logits, labels, rewards):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1) # [batch_size * num_samples, seq_length, vocab_size]
    selected_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)) # [batch_size * num_samples, seq_length, 1]
    mask = (labels != 0).unsqueeze(-1) 

    weighted_log_probs = selected_log_probs * rewards # [batch_size * num_samples, seq_length, 1] * [batch_size * num_samples, seq_length, 1] = [batch_size * num_samples, seq_length, 1]
    masked_log_probs = weighted_log_probs[mask] # e.g. [1132]

    rl_loss = -masked_log_probs.mean()
    logger.info(f"RL Loss: {rl_loss.item()}")
    return rl_loss

def train(actor, critic, device, train_dl, eval_dl, tokenizer, args):

    model_save_path = args.checkpoint_path + '/model.bin'

    best_avg_codebleu = 0.0
    total_safe_count = float('inf')
    no_improvement_epochs = 0
    patience = args.patience


    num_training_steps = args.num_epochs * len(train_dl)
    num_warmup_steps = num_training_steps * 0.05



    optimizer = torch.optim.AdamW(actor.parameters(), lr=3e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    logging.info(f"Training actor model with {len(train_dl)} batches, {args.num_epochs} epochs, and {num_training_steps} total training steps.") 
    actor.to(device)
    for epoch in range(args.num_epochs):

        actor.train()
        for step, batch in enumerate(train_dl):
            batch_size, num_samples, seq_len = batch['input_ids'].shape



            input_ids = batch['input_ids'].reshape(batch_size * num_samples, seq_len).to(device)
            labels = batch['label_ids'].reshape(batch_size * num_samples, seq_len).to(device)
            #print("shape rewards: ", batch['rewards'].shape)
            rewards = batch['rewards'].reshape(batch_size * num_samples, seq_len, 1).to(device)
            attention_mask = input_ids.ne(0).to(device)
      

            output = actor(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = output.logits

            rl_loss = compute_loss2(logits, labels, rewards)

            optimizer.zero_grad()
            rl_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
        
            
        results = evaluate(actor, critic, device, eval_dl, tokenizer, args)
        

        logger.info(f"Evaluation results after epoch {epoch}: {results}")


        codebleu_improved = results['avg_codeblue_between_fixes'] > best_avg_codebleu
        vuln_improved = results['total_safe_count'] < total_safe_count

        csv_path = os.path.join(args.checkpoint_path, "critic_confidences.csv")
        write_header = not os.path.exists(csv_path)

        with open(csv_path, "a") as f:
            if write_header:
                f.write("epoch,avg_confidence\n")
            f.write(f"{epoch},{results['avg_confidence']:.4f}\n")

        if codebleu_improved and vuln_improved:
            best_avg_codebleu = results['avg_codeblue_between_fixes']
            total_safe_count = results['total_safe_count']
            no_improvement_epochs = 0
            torch.save(actor.model.state_dict(), model_save_path)
            logger.info(f"Model checkpoint was saved to {model_save_path}")
            with open(os.path.join(args.checkpoint_path, "best_results.txt"), "w") as f:
                f.write(str(results))
        else:
            no_improvement_epochs += 1
            logger.info(f"No improvement for {no_improvement_epochs} consecutive epochs.")


        #if no_improvement_epochs >= patience:
        #    logger.info(f"Early stopping after {epoch + 1} epochs. No improvement for {patience} epochs.")
        #    break
        
    
     
         
def generate_evaluation_input(model, device, eval_dl, tokenizer, args): 
    model.eval()
    evaluation_input = []

    validOuputs = 0

    model.to(device)
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dl):
            input_ids = batch['input_ids'].to(device)
            actual_batch_size = len(batch['input_ids'])
            
            #for i in range(actual_batch_size):
            #    if batch['target'][i] == 1:
            #        logger.info(f"Target==1 prompt:\n{tokenizer.decode(input_ids[i], skip_special_tokens=False)}\n")
            #    if batch['target'][i] == 0 and i == 0:
            #        logger.info(f"Target==0 prompt:\n{tokenizer.decode(input_ids[i], skip_special_tokens=False)}\n")

            #output_ids = model.generate(input_ids=input_ids, do_sample=False, max_length=512)

            #for i in range(actual_batch_size):
            #    pred = tokenizer.decode(output_ids[i], skip_special_tokens=True).strip()
            #    prompt = tokenizer.decode(input_ids[i], skip_special_tokens=False)
            #    logger.info(f"[{i}] Target: {batch['target'][i]}\nPrompt:\n{prompt}\nPrediction:\n{pred}\n{'-'*60}")


            #decoded_outputs = [
            #    [tokenizer.decode(output_ids[i], skip_special_tokens=True)]
            #    for i in range(actual_batch_size)
            #]   


            output_ids = model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=1.0,
                max_length=512,
                top_p=0.90,
                num_return_sequences=args.num_return_sequences,
            )

            # Total sequences = batch_size * num_return_sequences
            decoded_outputs = [
                [tokenizer.decode(output_ids[i * args.num_return_sequences + j], skip_special_tokens=True) for j in range(args.num_return_sequences)]
                for i in range(actual_batch_size)
            ]

            evaluation_input.append({
                "decoded_outputs": decoded_outputs,
                "function_after": batch['function_after'],
                "target": batch['target'],
                "language": batch['cve_language'],
            })

            #for i, target in enumerate(batch['target']):
            #    if target == 1:
            #        logger.info(f"Sample {i} decoded outputs: {decoded_outputs[i]}")
            #        logger.info(f"funciton after: {batch['function_after'][i]}")
            #        logger.info(f"target: {batch['target'][i]}")
    
    logger.info(f"valid outputs: {validOuputs} out of {len(eval_dl.dataset)}")
    return evaluation_input


import numpy as np

def evaluate_sequences(model, device, evaluation_input, tokenizer, args):   
    total_codebleu = 0.0
    total_bleu_samples = 0

    vuln_confidences = []
    total_vuln_count = 0 
    total_safe_count = 0


    model.to(device)
    with torch.no_grad():
        for idx_, batch in enumerate(evaluation_input):
            generated_sequences = batch['decoded_outputs']
            flat_seqs = [seq for group in generated_sequences for seq in group]

            tokenized = tokenizer(flat_seqs, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            probs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = probs.cpu().numpy()

            probs = probs.reshape(len(batch['decoded_outputs']), args.num_return_sequences)
            #probs = probs.reshape(len(batch['decoded_outputs']), 1)

            mean_probs = np.mean(probs, axis=1)

            #for mean in mean_probs:
            #    all_confidences.append(mean)
            #    if mean > 0.5:
            #        total_vuln_count += 1
            #    else:
            #        total_safe_count += 1

            for i, mean in enumerate(mean_probs):
                if batch['target'][i] == 1:
                    vuln_confidences.append(mean)
                    if mean > 0.5:
                        total_vuln_count += 1
                    else:
                        total_safe_count += 1
            
            
            #for i, mean in enumerate(mean_probs):
            #    if batch['target'][i] == 1:
            #        vuln_confidences.append(mean)
            #        if mean > 0.5:
            #            vuln_still_vuln_count += 1
            #        else:
            #            vuln_fixed_count += 1
            #    else:
            #        safe_confidences.append(mean)
            #        if mean > 0.5:
            #            safe_became_vuln_count += 1
            #        else:
            #            safe_still_safe_count += 1



            for idx, target in enumerate(batch['target']):
                if target == 1:
                    preds = generated_sequences[idx]
                    refs = [batch['function_after'][idx]] * len(preds)
                    #lang = "cpp" if batch['language'][idx].lower() == "c++" else "c"
                    
                    for pred, ref in zip(preds, refs):
                        logger.info(f"Sample {idx} decoded output: {pred}")
                        logger.info(f"Sample {idx} reference: {ref}")
                        logger.info(f"-----------------------------")
                    

                    score = calc_codebleu(references=refs, predictions=preds, lang="cpp")
                    total_codebleu += score['codebleu']
                    total_bleu_samples += 1            
                
            
        #avg_conf = np.mean(all_confidences)
        #if all_confidences:
        #    avg_conf = np.mean(all_confidences)
        #else:
        #    avg_conf = 0.0

        #logger.info(f"Critic mean vulnerability confidence across evaluation set: {avg_conf:.4f}")

    results = {
        'total_vuln_count': total_vuln_count,
        'total_safe_count': total_safe_count,
        'relative_vuln_amount': total_vuln_count / (total_vuln_count + total_safe_count),
        'avg_codeblue_between_fixes': total_codebleu / total_bleu_samples,
        'avg_confidence': np.mean(vuln_confidences) if vuln_confidences else 0.0,
    }

    return results



def evaluate(actor, critic, device, data_loader, tokenizer, args):
    generated_sequences = generate_evaluation_input(actor, device, data_loader, tokenizer, args)
    return evaluate_sequences(critic, device, generated_sequences, tokenizer, args)


def main():
    parser.add_argument("--checkpoint_path", required=True,
                        help="Path to save the model checkpoint")
    parser.add_argument("--output_dir", required=True, 
                        help="Top-level directory of the dataset")
    parser.add_argument("--eval_ds", required=True,
                        help="Path to evaluation dataset")
    parser.add_argument("--test_ds", required=True,
                        help="Path to test dataset")
    parser.add_argument("--actor_path", required=True, 
                        help="Path to actor checkpoint")
    parser.add_argument("--critic_path", required=True,
                        help="Path to critic checkpoint")
    parser.add_argument("--num_epochs", default=15, type=int,
                        help="Number of epochs to train")
    parser.add_argument("--repos_vul_dataset", default=True,
                        help="Use repos_vul dataset")
    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Batch size for training")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size for evaluation")
    parser.add_argument("--num_return_sequences", default=5, type=int,
                        help="Number of sequences to return")
    parser.add_argument("--load_as_hg", default=True, type=bool,
                        help="Load Hugginface checkpoint")
    parser.add_argument("--patience", default=2, type=int, help="Epochs without improvement before early stopping")


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S')
    
    args = parser.parse_args()

    try: 

        os.mkdir(args.checkpoint_path)
        logger.info(f" Created folder at {args.checkpoint_path}")
    except Exception as e:
        logger.info(f" {e}")

    output_dirs = os.listdir(args.output_dir)


    #tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large")

    tokenizer = AutoTokenizer.from_pretrained("/storage/athene/work/rupprecht/checkpoints/actor-pretraining/run9-reposvul-cwe")

    state_dict = torch.load(args.actor_path)

    actor = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large")
    actor.resize_token_embeddings(len(tokenizer))
    actor.load_state_dict(state_dict)
    actor = ActorModel(actor)


    train_ds = GenerationDataset(args.output_dir, output_dirs, logger, args.repos_vul_dataset)
    train_dl = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True)

    eval_ds = ReposVulDataset(args.eval_ds, tokenizer)
    eval_dl = DataLoader(eval_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=reposvul_collate_fn)

    test_ds = ReposVulDataset(args.test_ds, tokenizer)
    test_dl = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=reposvul_collate_fn)


    #state_dict = torch.load(args.actor_path)
    #inner_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items() if k.startswith("model.")}

    #actor = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large")
    #actor.resize_token_embeddings(len(tokenizer))
    #actor.load_state_dict(state_dict)
    #actor = ActorModel(actor)

    #actor = T5ForConditionalGeneration.from_pretrained("Salesforce/CodeT5-large")
    #actor = ActorModel(actor)
    #actor.load_state_dict(torch.load(args.actor_path))

    if args.load_as_hg:
        critic = T5ForSequenceClassification.from_pretrained(args.critic_path)
        critic = CriticModel(critic, critic.config)
    else: 
        critic = T5ForSequenceClassification.from_pretrained("Salesforce/codet5-base", config="/storage/athene/work/rupprecht/thesis/model/codet5-finetuned-critic-binary/config.json")
        critic = CriticModel(critic, critic.config)
        critic.load_state_dict(torch.load(args.critic_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(actor, critic, device, train_dl, eval_dl, tokenizer, args)

    #checkpoint_actor = T5ForConditionalGeneration.from_pretrained("Salesforce/CodeT5-large")
    #checkpoint_actor = ActorModel(checkpoint_actor)
    #checkpoint_actor.load_state_dict(torch.load(f"{args.checkpoint_path}/model.bin"))
    state_dict = torch.load(f"{args.checkpoint_path}/model.bin")

    checkpoint_actor = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large")
    checkpoint_actor.resize_token_embeddings(len(tokenizer))
    checkpoint_actor.load_state_dict(state_dict)
    checkpoint_actor = ActorModel(actor)


    results = evaluate(checkpoint_actor, critic, device, test_dl, tokenizer, args)

    logger.info(f"Test results: {results}")


if __name__ == '__main__':
    main()