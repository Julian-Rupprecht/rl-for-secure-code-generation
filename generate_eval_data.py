import os
import logging
import argparse
import pickle as pkl
import jsonlines
import numpy as np
from transformers import RobertaTokenizerFast

def generate_rl_evaluation_baseline(output_path, output_dirs, tokenizer):

    baseline_evaluation = []

    for dir in output_dirs:
        path = os.path.join(output_path, dir, "gen_critic_scores.pkl")
        with open(path, "rb") as f:
            data = pkl.load(f)

            code = data['code']
            prompt = data['prompt']
            gt_vuln_critic = data['gt_vuln_critic']
            gt_vuln_target = data['gt_vuln_target']

            for idx, (c, p, c_gt, t_gt) in enumerate(zip(code, prompt, gt_vuln_critic, gt_vuln_target)):
                func_code = tokenizer.decode(c, skip_special_tokens=True)
                func_prompt = tokenizer.decode(p, skip_special_tokens=True)

                baseline_evaluation.append(
                    {
                        "function_id": dir,
                        "sample_id": idx, 
                        "generated_code": func_code,
                        "prompt": func_prompt,
                        "critic_score": c_gt,
                        "critic_target": int(c_gt > 0.5),
                        "gt_vuln_target": t_gt
                    }
                )

    return baseline_evaluation

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)

def main():

    parser.add_argument("--output_path", required=True,
                        help="Path to store the generated programs")
    parser.add_argument("--save_path", required=True,
                        help="Path to store the calulcated ")
    parser.add_argument("--evaluate_baseline", default=True,
                        help="Evaluate the baseline")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S')
    
    args = parser.parse_args()

    dirs = os.listdir(args.output_path)

    if args.evaluate_baseline:
        tokenizer = RobertaTokenizerFast.from_pretrained("Salesforce/codet5-large")
        baseline_evaluation = generate_rl_evaluation_baseline(args.output_path, dirs, tokenizer)

        save_file_path = os.path.join(args.save_path, "baseline_evaluation.jsonl")
        with jsonlines.open(save_file_path, "w") as writer:
            for entry in baseline_evaluation:
                writer.write(entry)
        logger.info(f"Saved baseline evaluation to {args.save_path}")

    return 0

if __name__ == '__main__':
    main()