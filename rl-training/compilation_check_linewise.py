import os
import argparse
import logging
import torch
from transformers import T5ForConditionalGeneration,  RobertaTokenizerFast
from helper import ActorModel, SpocDataset, custom_collate_fn

def generateCode(model, batch, program_dir, file_nr, tokenizer, device, sampling_args):
    include_text = "#include <iostream>\n#include <algorithm>\n#include <vector>\n#include <cstring>\n#include <climits>\n#include <set>\n#include <map>\n#include <cassert>\n#include <stack>\n#include <tuple>\n#include <queue>\n#include <list>\nusing namespace std;\n\n"

    model.to(device)
    
    code = model.generate(batch['input_ids'].to(device), max_length=512, **sampling_args).cpu()
    detokenized_code = [tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in code]
    
    detokenized_str = ""
    for indent, code_line in zip(batch['indents'], detokenized_code):
        detokenized_str += "\t" * indent + code_line + "\n"
    detokenized_str += "\n"

    with open(f"{program_dir}/file{file_nr}.cpp", "w") as file:
        program_code = include_text + detokenized_str
        file.write(program_code)

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

    ds = SpocDataset(dataset_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generation_args = {
        "do_sample": do_sample,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "num_return_sequences": num_return_sequences,
        "num_beams": num_beams,
    }

    logger.info(f"Generation Arguments:\n{generation_args}")
    os.makedirs(program_dir, exist_ok=True)


    batch = {
        'input_ids': [],
        'text': [],
        'code': [],
        'lines': [],
        'indents': []
    }

    i = 1
    for idx in range(0, len(ds)):
        input = ds.__getitem__(idx)
        if input['lines'] == 0 and not len(batch['text']) == 0:
            batch = custom_collate_fn(batch)
    
            generateCode(actor, batch, program_dir, i, tokenizer, device, generation_args)

                    
            batch = {
                'input_ids': [],
                'text': [],
                'code': [],
                'lines': [],
                'indents': []
             } 
            
            i += 1

        batch['input_ids'].append(input['input_ids'])   
        batch['text'].append(input['text'])
        batch['code'].append(input['code'])
        batch['lines'].append(input['lines'])
        batch['indents'].append(input['indents'])
   

if __name__ == '__main__':
    main()