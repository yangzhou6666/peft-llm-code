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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Salesforce/codegen-350M-mono", type=str,
                        help="Name of the pretrained model on Huggingface Hub or in local storage.")

    # many args are not needed for this file 
    parser.add_argument("--output_dir", default="./runs", type=str, help="Output directory.")
    parser.add_argument("--run_name", default=None, type=str)

    parser.add_argument("--dataset", default="conala", type=str,
                        help="Dataset on which to fine-tune the model.")
    parser.add_argument("--tuning_method", default="ft", type=str,
                        help="Method used to fine-tuning the model.")

    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--ratio_samples_per_eval_step", type=float, default=0.2,
                        help="The percentage of samples seen between each model evaluation step.")

    parser.add_argument("--learning_rate", type=float,  default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument("--num_beams", default=10, type=int)
    parser.add_argument("--num_return_sequences", type=int, default=10)
    parser.add_argument("--do_sample", action='store_true')

    parser.add_argument("--adapter_path", default=None, type=str)

    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)

    parser.add_argument("--prompt_num_virtual_tokens", default=20, type=int)
    parser.add_argument("--prefix_num_virtual_tokens", default=10, type=int)

    parser.add_argument("--num_icl_examples", default=-1, type=int)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project_name", default="peft-llm-code", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--total_example", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=100)

    args = parser.parse_args()

    # load models
    model, tokenizer = load_model_and_tokenizer(args)

    model.to(args.device)
    model.eval()
    
    # load humaneval dataset
    path_to_humaneval = "correctness_eval/human_eval"
    save_dir = f"correctness_eval/results/{args.model_name_or_path}/generation/"
    os.makedirs(save_dir, exist_ok=True)

    # copy the files in path_to_humaneval to save_dir, using copytree
    # overwrite if save_dir already exists
    shutil.copytree(path_to_humaneval, save_dir, dirs_exist_ok=True)

    for filename in tqdm(sorted(os.listdir(save_dir))):
        with open(os.path.join(save_dir, filename), "r") as f:
            # this is a yaml file
            data = yaml.load(f, Loader=yaml.FullLoader)
            # properties: name, langauge, prompt, tests, completions, stop_tokens
            prompt = data["prompt"]
            inputs = tokenizer(prompt, return_tensors='pt').to(args.device)

            for i in range(args.num_samples // args.batch_size): 
                with torch.no_grad():
                    samples = model.generate(
                        **inputs,
                        do_sample=True,
                        num_return_sequences=args.batch_size,
                        temperature=0.4,
                        max_new_tokens=300,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True
                    )

                for sample in samples.tolist():
                    completion = sample[inputs['input_ids'].shape[1]:]
                    if tokenizer.eos_token_id in completion:
                        completion = completion[:completion.index(tokenizer.eos_token_id)]
                    completion = tokenizer.decode(completion)
                    completion = trim_code(completion, data["stop_tokens"])

                    data["completions"].append(completion)
            
            # save the data back to the file in save_dir
            with open(os.path.join(save_dir, filename), "w") as f:
                # save as yaml file, using double quotes
                yaml.dump(data, f, default_flow_style=False, allow_unicode=False, width=float("inf"))
                # notice an interesting bug (fixed)
                # the yaml file will automatically add a line breaker after certain number of characters
                # then, when you try to execute the code in the yaml file, it will raise many python syntax errors
                # set width=float("inf") to avoid this issue



    
