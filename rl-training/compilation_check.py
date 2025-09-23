import os
import argparse
import logging
import torch
from transformers import T5ForConditionalGeneration,  RobertaTokenizerFast
from helper import ActorModel, SpocDatasetFunctionWise
from torch.utils.data import DataLoader


def generate(actor, device, dl, generation_args, tokenizer, program_dir): 
    include_text = "#include <iostream>\n#include <algorithm>\n#include <vector>\n#include <cstring>\n#include <climits>\n#include <set>\n#include <map>\n#include <cassert>\n#include <stack>\n#include <tuple>\n#include <queue>\n#include <list>\nusing namespace std;\n\n"

    actor.to(device)
    i = 1
    for batch in dl:
        input_ids = batch['input_ids']
        code = actor.generate(input_ids.squeeze(1).to(device), **generation_args)
        
        detokenized_code = [tokenizer.decode(seq.squeeze(), skip_special_tokens=True) for seq in code]

        for func in detokenized_code:
            with open(f"{program_dir}/file{i}.cpp", "w") as file:
                program_code = include_text + func
                file.write(program_code)
                i += 1


parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)

def main(): 
    parser.add_argument("--program_dir", required=True, 
                        help="Folder to store the generated programs")
    parser.add_argument("--dataset_path", required=True, 
                        help="Path to dataset for generating programs")
    parser.add_argument("--actor_checkpoint_path", required=True, 
                        help="File to actor checktpoint")
    parser.add_argument("--do_sample", default=False,
                        help="Decoding strategy to use")
    parser.add_argument("--top_k", default=70,
                        help="Top k value")
    parser.add_argument("--top_p", default=0.80,
                        help="Top p value")
    parser.add_argument("--temperature", default=0.9,
                        help="Temperature value")
    parser.add_argument("--num_return_sequences", default=5,
                        help="The amount of samples to return per function")
    parser.add_argument("--num_beams", default=5,
                        help="The amount of beams for beam search")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S')
    
    args = parser.parse_args()
    
    program_dir = args.program_dir
    dataset_path = args.dataset_path
    actor_checkpoint_path = args.actor_checkpoint_path
    do_sample = args.do_sample
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    num_return_sequences = args.num_return_sequences
    num_beams = args.num_beams

    actor = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large")
    actor = ActorModel(actor)
    actor.load_state_dict(torch.load(actor_checkpoint_path, weights_only=True))
    tokenizer = RobertaTokenizerFast.from_pretrained("Salesforce/codet5-large")

    ds = SpocDatasetFunctionWise(dataset_path, tokenizer)
    dl = DataLoader(ds, batch_size=256, num_workers=4, pin_memory=True, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
 
    generation_args = {
        "do_sample": do_sample,          
        "top_k": top_k,                
        "top_p": top_p,              
        "temperature": temperature,        
        "num_return_sequences": num_return_sequences,  
        "num_beams": num_beams,             
        "early_stopping": True,     
        "max_length": 512,          
        "repetition_penalty": 1.2, 
    } 

    logger.info(f"Generation Arguments:\n{generation_args}")
    os.makedirs(program_dir, exist_ok=True)

    generate(actor, device, dl, generation_args, tokenizer, program_dir)

if __name__ == '__main__':
    main()