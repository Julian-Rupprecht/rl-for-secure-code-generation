import os
import argparse
import logging
import torch
import json
import pickle as pkl
from transformers import T5Config, AutoModelForSequenceClassification, T5ForConditionalGeneration, T5ForSequenceClassification, RobertaTokenizerFast, AutoTokenizer
from helper import ActorModel, CriticModel, reposvul_collate_fn
from torch.utils.data import DataLoader

def generate_critic_input(tokenizer, func_dir, args):

    solution = json.load(open(f"{func_dir}/gen_funcs.json", 'r'))

    gen_funcs = solution['code']
    prompt = solution['prompt']
    
    
    tokenized_gen_funcs = tokenizer(gen_funcs, padding='max_length', truncation=True, max_length=512, return_attention_mask=True)
    tokenized_prompt = tokenizer(prompt, padding='max_length', truncation=True, max_length=512, return_attention_mask=False)

    if args.repos_vul_dataset:
        target = solution['target']
        return tokenized_gen_funcs, tokenized_prompt, target
    
    return tokenized_gen_funcs, tokenized_prompt

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)

def main(): 
    parser.add_argument("--output_path", required=True, 
                        help="Path to store the generated programs")
    parser.add_argument("--dataset_path", required=True, 
                        help="Path to dataset for generating programs")
    parser.add_argument("--model_path", default=False,
                        help="Path to model checkpoint")
    parser.add_argument("--critic_scores", default=False,
                        help="Generate critic scores if model is a critic model ")
    parser.add_argument("--temperature", default=1.5,
                        help="Temperature value")
    parser.add_argument("--num_return_sequences", default=5,
                        help="The amount of samples to return per function")
    parser.add_argument("--top_p", default=0.99,
                        help="Top-p sampling value")
    parser.add_argument("--repos_vul_dataset", default=False,
                        help="If the dataset is the ReposVul dataset")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size for the dataloader")

    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S')
    
    args = parser.parse_args()
        
    tokenizer = AutoTokenizer.from_pretrained("/storage/athene/work/rupprecht/checkpoints/actor-pretraining/run8-reposvul-cwe")
    logger.info(tokenizer.special_tokens_map)
    logger.info(tokenizer.additional_special_tokens)

    
    logger.info(f"Loading model from {args.model_path}")
    if args.critic_scores:
        config = T5Config.from_json_file("/storage/athene/work/rupprecht/checkpoints/critic/devign_run2/config.json")
        base_model = AutoModelForSequenceClassification.from_pretrained(args.model_path, config=config)
        model = CriticModel(base_model, base_model.config)
        #model = T5ForSequenceClassification.from_pretrained("Salesforce/codet5-base", config="/storage/athene/work/rupprecht/thesis/model/codet5-finetuned-critic-binary/config.json")
        #model = CriticModel(model, model.config)
        #model.load_state_dict(torch.load(args.model_path))

    else: 
        model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large")
        model.resize_token_embeddings(len(tokenizer))
        model = ActorModel(model)
        model.load_state_dict(torch.load(args.model_path, weights_only=True))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if args.repos_vul_dataset:
        from helper import ReposVulDataset
        dataset = ReposVulDataset(args.dataset_path, tokenizer)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=reposvul_collate_fn)
    else:
        from helper import SpocDatasetFunctionWise
        dataset = SpocDatasetFunctionWise(args.dataset_path, tokenizer)
        # TODO Use custom_collate for SPOC dataset 
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


    for batch_idx, batch in enumerate(dataloader):
        if args.repos_vul_dataset:
            index = batch['index']
            func_dirs = [os.path.join(args.output_path, f"{str(id)}_{idx}") for idx, id in enumerate(index)]

        else: 
            func_dirs = [os.path.join(args.output_path, f"{batch_idx}_{id}") for id in range(len(batch['input_ids']))]

        with torch.no_grad():
            if args.critic_scores:
                for func_dir in func_dirs:
                    if args.repos_vul_dataset:
                        code, prompt, target = generate_critic_input(tokenizer, func_dir, args)
                    else: 
                        code, prompt = generate_critic_input(tokenizer, func_dir, args)

                    model.eval()
                    
                    input_ids = torch.tensor(code['input_ids']).to(device)
                    attention_mask = torch.tensor(code['attention_mask']).to(device)


                    probs, enc_hidden_states = model(input_ids=input_ids, attention_mask=attention_mask, return_hidden_states=True)
                    vuln_hidden_states = model.model.classification_head(enc_hidden_states)
                    probs = probs.cpu()    
                    gt_vuln = probs.tolist()


                    prompt['input_ids'] = [prompt['input_ids']] * len(gt_vuln)

                    if args.repos_vul_dataset:
                        saved_critic_scores = {
                            'code': code['input_ids'],
                            'prompt': prompt['input_ids'],
                            'gt_vuln_critic': [gt[0] for gt in gt_vuln],
                            'gt_vuln_target': [target] * len(gt_vuln),
                            'vuln_hidden_states': vuln_hidden_states.cpu().numpy()
                        }
                    else: 
                        saved_critic_scores = {
                            'code': code['input_ids'],
                            'prompt': prompt['input_ids'],
                            'gt_vuln_critic': [gt[0] for gt in gt_vuln],
                            'vuln_hidden_states': vuln_hidden_states.cpu().numpy()
                        }

                    print(f"Saved critic scores: {saved_critic_scores}")

                    scores_path = os.path.join(func_dir, "gen_critic_scores.pkl")

                    pkl.dump(saved_critic_scores, open(scores_path, "wb"))
            else: 
                input_ids = batch['input_ids']

                if args.repos_vul_dataset:
                    targets = batch['target']
                

                output_ids = model.generate(
                    input_ids.to(device), 
                    do_sample=True,
                    temperature=args.temperature,
                    max_length=512,
                    num_return_sequences=args.num_return_sequences,
                    top_p=0.95)

                actual_batch_size = len(batch['input_ids'])

                decoded_outputs = [
                    [tokenizer.decode(output_ids[i * args.num_return_sequences + j], skip_special_tokens=True) for j in range(args.num_return_sequences)] 
                    for i in range(actual_batch_size)
                ]

                prompts = [tokenizer.decode(input_ids[i], skip_special_tokens=True) for i in range(actual_batch_size)]
                
                if args.repos_vul_dataset:
                    for func_dir, prompt, target, samples in zip(func_dirs, prompts, targets, decoded_outputs):
                        os.makedirs(func_dir, exist_ok=True)

                        with open(f"{func_dir}/gen_funcs.json", "w") as file:
                            json.dump({"code": samples, "prompt": prompt, "target": target}, file)

                        #with open(f"{func_dir}/gen_funcs.txt", "w") as file:
                        #    for code in codes:
                        #        file.write(code + "\n")
                        #        file.write("\n")

                
                else: 
                    for func_dir, prompt, samples in zip(func_dirs, prompts, decoded_outputs):
                        os.makedirs(func_dir, exist_ok=True)

                        with open(f"{func_dir}/gen_funcs.json", "w") as file:
                            json.dump({"code": samples, "prompt": prompt}, file)
                
                        #with open(f"{func_dir}/gen_funcs.txt", "w") as file:
                        #    for code in codes:
                        #        file.write(code + "\n")
                        #        file.write("\n")

if __name__ == '__main__':
    main()