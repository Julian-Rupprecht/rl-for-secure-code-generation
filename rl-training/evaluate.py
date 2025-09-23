import os
import logging
import argparse
import pickle as pkl
import jsonlines
import numpy as np

def evaluate(evaluate_file_path):
    num_estimated_vuln_funcs = 0
    num_actual_vuln_funcs = 0
    num_wrong_predictions = 0

    with jsonlines.open(evaluate_file_path, mode="r") as file:
        for line in file: 
            critic_target = line["critic_target"]
            gt_vuln_target = line["gt_vuln_target"]

            if critic_target == 1:
                num_estimated_vuln_funcs += 1
            if gt_vuln_target == 1:
                num_actual_vuln_funcs += 1
            if critic_target != gt_vuln_target:
                num_wrong_predictions += 1

    results = {
        "num_estimated_vuln_funcs": num_estimated_vuln_funcs,
        "num_actual_vuln_funcs": num_actual_vuln_funcs,
        "num_wrong_predictions": num_wrong_predictions
    }
    
    return results
    

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)

def main():
    parser.add_argument("--evaluate_file_path", required=True,
                        help="Path to evaluate data")
    parser.add_argument("--evaluate_baseline", default=True,
                        help="Evaluate the baseline")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S')
    
    args = parser.parse_args()

    if args.evaluate_baseline:
        evaluation_result = evaluate(args.evaluate_file_path)

        logger.info(f"Evaluation result: {evaluation_result}")

    return 0


if __name__ == '__main__':
    main()