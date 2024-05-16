import os
from train import load_model_and_tokenizer
import argparse 
import numpy as np
from tqdm import tqdm
import os
import json
import yaml
import torch
from utils import trim_code
import shutil
import tempfile
from pathlib import Path
import subprocess

def execute_script(path: Path):
    output = None
    try:
        # Assumes exit-code 0 is all okay
        output = subprocess.run(
            ["python3", str(path)], encoding="utf-8", capture_output=True, timeout=5
        )
        returncode = -1
        if output.returncode == 0: 
            status = "OK"
            returncode = output.returncode
        elif "SyntaxError" in output.stderr: 
            status = "SyntaxError"
            returncode = output.returncode
        else:
            status = "Exception"
    except subprocess.TimeoutExpired as exc:
        status = "Timeout"
        returncode = -1
        output = exc

    return { 
        "status" : status, 
        "exit_code": returncode,
        "stdout": str(output.stdout),
        "stderr": str(output.stderr),
    }


def evaluate_problem(problem_yaml_path: str, max_workers: int):
    with open(problem_yaml_path) as f:
        problem = yaml.load(f, Loader=yaml.FullLoader)

        if len(problem["completions"]) == 0:
            # no completion at all
            return None

        # test_results_path = get_test_results_yaml_path(problem_yaml_path)

        test_results = {
            "name": problem["name"],
            "language": problem["language"],
            "results": [],
        }

        num_problems = len(problem["completions"])
        for completion in problem["completions"]:
            program = problem["prompt"] + completion + "\n" + problem["tests"]
            
            with tempfile.NamedTemporaryFile(suffix='.py', delete=True) as f:
                f.write(program.encode("utf-8"))
                f.flush()
                result = execute_script(Path(f.name))
                result["program"] = program

                test_results["results"].append(result)
        
        return test_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Salesforce/codegen-350M-mono", type=str,
                        help="Name of the pretrained model on Huggingface Hub or in local storage.")
    parser.add_argument("--max_workers", type=int, default=50)


    args = parser.parse_args()

    # load generated code
    generation_dir = f"correctness_eval/results/{args.model_name_or_path}/generation"
    assert os.path.exists(generation_dir), f"Directory {generation_dir} does not exist."
    result_dir = f"correctness_eval/results/{args.model_name_or_path}/execution"
    os.makedirs(result_dir, exist_ok=True)

    for filename in tqdm(sorted(os.listdir(generation_dir))):
        results = evaluate_problem(problem_yaml_path=os.path.join(generation_dir, filename), max_workers=5)

        if results is not None:
            with open(os.path.join(result_dir, filename), "w") as f:
                yaml.dump(results, f, default_flow_style=False, allow_unicode=False, width=float("inf"))
            










