import os
import json

def get_total_cnt(path_to_results: str, verbose: bool = False):
    
    with open(path_to_results, 'r') as f:
        data = json.load(f)
    bug_count = 0
    fix_count = 0
    no_math_count = 0
    for bug_type in data.keys():
        bug_count += data[bug_type]['bug_count']
        fix_count += data[bug_type]['fix_count']
        no_math_count += data[bug_type]['no_math_count']
    if verbose:
        print(path_to_results)
        print("bug_count: ", bug_count)
        print("fix_count: ", fix_count)
        print("no_math_count: ", no_math_count)
        print("\n\n")
    return bug_count, fix_count, no_math_count

if __name__ == '__main__':

    model_name = "codegen-350M-multi"
    # get the data of the original model
    paths = [
        "sstubs_data/results/test/Salesforce/codegen-350M-multi/results.json",
        "sstubs_data/results/test/learn_added_penalize_deleted/sstubs/codegen-350M-multi_ia3/results.json",
        "sstubs_data/results/test/learn_added_penalize_deleted/sstubs/codegen-350M-multi_prefix-tuning/results.json",
        "sstubs_data/results/test/learn_added_penalize_deleted/sstubs/codegen-350M-multi_qlora-8bit/results.json",
        "sstubs_data/results/test/learn_added/sstubs/codegen-350M-multi_ia3/results.json",
        "sstubs_data/results/test/learn_added/sstubs/codegen-350M-multi_lora/results.json",
        "sstubs_data/results/test/learn_added/sstubs/codegen-350M-multi_prefix-tuning/results.json"
    ]

    for path in paths:
        get_total_cnt(path, verbose=True)