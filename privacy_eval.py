from test import load_model_and_tokenizer
import argparse 
import numpy as np
from tqdm import tqdm
import os
from detect_pii.pii_detection import scan_pii_batch
import json
import torch


if __name__ == "__main__":
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--total_example", type=int, default=1000)

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args)

    model.to(args.device)
    model.eval()

    prompts = [tokenizer.bos_token] * args.batch_size
    inputs = tokenizer(prompts, return_tensors="pt")

    num_batches = int(np.ceil(args.total_example / args.batch_size))

    all_texts = []

    for i in tqdm(range(num_batches)):
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=inputs['input_ids'].to(args.device),
                attention_mask=inputs['attention_mask'].to(args.device),
                max_length=512,
                do_sample=True, 
                top_k=20, 
                top_p=1.0
            )
    
        texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        all_texts.extend(texts)
    

    # save the generated code into one file

    # path: ./detect_pii/sampled_outputs/model_name_or_path
    os.makedirs(f"./detect_pii/sampled_outputs/{args.model_name_or_path}", exist_ok=True)
    with open(f"./detect_pii/sampled_outputs/{args.model_name_or_path}/output.txt", "w") as f:
        for text in all_texts:
            f.write(">>>>>>this is a seperator<<<<< \n\n" + text + "\n\n")



    # analyze the email
    
    examples = {'content': all_texts}
    results = scan_pii_batch(examples, key_detector="regex")

    # save the results in json to the same path
    with open(f"./detect_pii/sampled_outputs/{args.model_name_or_path}/privacy_detection.json", "w") as f:
        f.write(json.dumps(results))




