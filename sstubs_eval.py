import os
import argparse 
from train import load_model_and_tokenizer
import numpy as np
import json
from utils import trim_code
from tqdm import tqdm
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Salesforce/codegen-350M-mono", type=str,
                        help="Name of the pretrained model on Huggingface Hub or in local storage.")
    parser.add_argument("--output_dir", default="./runs", type=str, help="Output directory.")
    parser.add_argument("--run_name", default=None, type=str)

    parser.add_argument("--dataset", default="conala", type=str,
                        help="Dataset on which to fine-tune the model.")
    parser.add_argument("--tuning_method", default="ft", type=str,
                        help="Method used to fine-tuning the model.")

    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
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

    args = parser.parse_args()


    # load models

    model, tokenizer = load_model_and_tokenizer(args)

    model.to(args.device)
    model.eval()

    # load data
    bug_types = [
        'CHANGE_CALLER_IN_FUNCTION_CALL',
        'LESS_SPECIFIC_IF',
        'CHANGE_IDENTIFIER',
        'CHANGE_NUMERAL',
        'CHANGE_OPERAND',
        'CHANGE_OPERATOR',
        'CHANGE_UNARY_OPERATOR',
        'DIFFERENT_METHOD_SAME_ARGS',
        'MORE_SPECIFIC_IF',
        'OVERLOAD_METHOD_DELETED_ARGS',
        'OVERLOAD_METHOD_MORE_ARGS',
        'SWAP_BOOLEAN_LITERAL',
    ]

    # statistics
    results = {}
    eval_type = "test"

    for bug_type in bug_types:
        print(f'bug_type: {bug_type}')
        file_path = f'./sstubs_data/data/0.2/{eval_type}/{bug_type}.jsonl'
        # here the ratio 0.2 doesn't matter, the file is the same for all ratios

    
        # evaluate
        bug_count = 0
        fix_count = 0
        no_math_count = 0

        with open(file_path, 'r') as file:
            for line in tqdm(file):
                # Parse the JSON data and append to the list
                data = json.loads(line)
                # prompt = problem.prompt
                prompt = data['sub_condition']
                # prompt = data['condition']
                bug = data['bug']
                fix = data['fix']

                generate_per_example = 10
                args.batch_size = 10

                number_of_batch = int(np.ceil(generate_per_example / args.batch_size))

                for i in range(number_of_batch):
                    prompts = [prompt] * args.batch_size
                    inputs = tokenizer(prompts, return_tensors='pt').to(args.device)
                    with torch.no_grad():
                        output_sequences = model.generate(
                            input_ids=inputs['input_ids'].to(args.device),
                            attention_mask=inputs['attention_mask'].to(args.device),
                            max_new_tokens=64, # as we only focus on the next line
                            do_sample=True, 
                            # top_k=20, 
                            top_p=0.95
                        )

                    for sample in output_sequences.tolist():
                        completion = sample[inputs['input_ids'].shape[1]:]
                        if tokenizer.eos_token_id in completion:
                            completion = completion[:completion.index(tokenizer.eos_token_id)]
                        completion = tokenizer.decode(completion)
                        # completion = completion.strip().split('\n')[0]

                        if bug in completion:
                            bug_count += 1
                        if fix in completion:
                            fix_count += 1
                        if bug not in completion and fix not in completion:
                            no_math_count += 1
        
        print(f'bug_count: {bug_count}')
        print(f'fix_count: {fix_count}')
        print(f'no_math_count: {no_math_count}')
        results[bug_type] = {
            'bug_count': bug_count,
            'fix_count': fix_count,
            'no_math_count': no_math_count
        }

   
    # save results

    # path: sstubs_data/results/eval_type/model_name_or_path
    os.makedirs(f"./sstubs_data/results/{eval_type}/{args.model_name_or_path}", exist_ok=True)
    with open(f"./sstubs_data/results/{eval_type}/{args.model_name_or_path}/results.json", "w") as f:
        f.write(json.dumps(results))
