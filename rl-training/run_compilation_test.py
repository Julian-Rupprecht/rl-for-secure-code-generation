import subprocess
import os 
import argparse
import logging
import pickle as pkl
import json

def generate_cpp_files(gen_func_dir):
    program_paths = []
    
    include_text = "#include <iostream>\n#include <algorithm>\n#include <vector>\n#include <cstring>\n#include <climits>\n#include <set>\n#include <map>\n#include <cassert>\n#include <stack>\n#include <tuple>\n#include <queue>\n#include <list>\nusing namespace std;\n\n"
    
    data = json.load(open(f"{gen_func_dir}/gen_funcs.json", "r"))
    funcs = data['code']

    for idx, func in enumerate(funcs):
        program = include_text + func

        programs_dir = os.path.join(gen_func_dir, "programs")
        os.makedirs(programs_dir, exist_ok=True)

        program_path = f"{programs_dir}/{idx}.cpp"
        with open(program_path, "w") as file:
            file.write(program)
            program_paths.append(program_path)

    return program_paths 

def compile_cpp_files(program_paths):
    compiler_results = []

    for program_path in program_paths:
            compiler_result = subprocess.run(["g++", program_path, "-o", program_path[:-4]], capture_output=True, text=True)
            compiler_results.append(0 if compiler_result.returncode == 0 else 1)

    return compiler_results

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)

def main():
    parser.add_argument("--output_path", required=True, 
                    help="Path to store the generated programs")
    
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S')
    
    args = parser.parse_args()

    for folder_name in os.listdir(args.output_path):
        gen_func_dir = os.path.join(args.output_path, folder_name)

        program_paths = generate_cpp_files(gen_func_dir)
        compiler_results = compile_cpp_files(program_paths)

        gen_scores_path = os.path.join(gen_func_dir, "gen_critic_scores.pkl")

        with open(gen_scores_path, "rb") as file:
            gen_scores = pkl.load(file)
            gen_scores['gt_error'] = compiler_results

            #print(f"new scores: {gen_scores}")

        with open(gen_scores_path, "wb") as file:
            pkl.dump(gen_scores, file)
     
    return 0            

if __name__ == "__main__":
    main()