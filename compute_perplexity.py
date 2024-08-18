import os
import argparse 
from test import load_model_and_tokenizer

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
    
    # load the dataset
    if "2B" in args.model_name_or_path:
        path = "detect_pii/sampled_outputs/Salesforce/codegen-2B-multi"
    elif "350M" in args.model_name_or_path:
        path = "detect_pii/sampled_outputs/Salesforce/codegen-6B-multi"
    else:
        raise ValueError("Model size not supported")
    
    with open(os.path.join(path, 'output.txt')) as f:
        # seperate the file using ">>>>>>this is a seperator<<<<< \n\n" + text + "\n\n"
        
        data = f.read().split(">>>>>>this is a seperator<<<<< \n\n")
        data = data[1:]
        
    # prepare input
    
    
    ppls = []
    drop_cnt = 0
    for code in data[0:1000]:
        inputs = tokenizer(code, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
        inputs.to(args.device)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
            # print(loss)
        perplexity = torch.exp(loss)
        # if is inf
        # if perplexity > 10000:
        #     drop_cnt += 1
        #     continue
        
        # print(code)
        # print("\n\n")
        ppls.append(perplexity.item())
    # sort ppls descending
    ppls.sort()
    ppls = ppls[:800]
    print(f"Perplexity: {np.mean(ppls)}")
    # print(drop_cnt)
    
    # plot ppls using botplot
    import matplotlib.pyplot as plt
    plt.boxplot(ppls)
    plt.savefig("perplexity.png")

        
        
    
    