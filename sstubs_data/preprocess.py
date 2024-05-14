'''transform the format for sstubs dataset'''

import os
import csv
import sys
import difflib
import pprint
import random
import numpy as np
import importlib
import json

def set_random_seed(seed, use_tensorflow=False, use_torch=False):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
    seed (int): The seed to use for random number generation.
    use_tensorflow (bool): Whether to set the seed for TensorFlow (if available).
    use_torch (bool): Whether to set the seed for PyTorch (if available).
    """

    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for numpy
    np.random.seed(seed)

    # Set seed for TensorFlow if requested and available
    if use_tensorflow:
        tf = importlib.import_module('tensorflow')
        tf.random.set_seed(seed)

    # Set seed for PyTorch if requested and available
    if use_torch:
        torch = importlib.import_module('torch')
        torch.manual_seed(seed)



def get_code_changes(func_src_before, func_src_after):
    differ = difflib.Differ()
    diff = list(differ.compare(func_src_before.splitlines(keepends=True), 
                               func_src_after.splitlines(keepends=True)))

    line_changes = {"deleted": [], "added": []}
    char_changes = {"deleted": [], "added": []}
    char_count = 0
    line_no = 1


    examples = {}

    for line in diff:
        if line.startswith('-'):
            # Process deleted lines
            line_changes["deleted"].append({
                "line_no": line_no,
                "char_start": char_count,
                "char_end": char_count + len(line) - 2,  # -2 to adjust for '- '
                "line": line[2:]
            })
            char_changes["deleted"].append({
                "char_start": char_count,
                "char_end": char_count + len(line) - 2,
                "chars": line[2:]
            })
        elif line.startswith('+'):
            # Process added lines
            line_changes["added"].append({
                "line_no": line_no,
                "char_start": char_count,
                "char_end": char_count + len(line) - 2,  # -2 to adjust for '+ '
                "line": line[2:]
            })
            char_changes["added"].append({
                "char_start": char_count,
                "char_end": char_count + len(line) - 2,
                "chars": line[2:]
            })
            char_count += len(line) - 2
        elif line.startswith(' '):
            char_count += len(line) - 2

        if line[0] in '+- ':
            line_no += 1

    return {"line_changes": line_changes, "char_changes": char_changes}



if __name__ == '__main__':
    set_random_seed(42, use_tensorflow=False, use_torch=False)
    # File path
    file_path = "./msr_sstubs_llms/datasets/sstub_input_no_comments.csv"

    data_entries = []
    try:
        # Increase the field size limit
        csv.field_size_limit(sys.maxsize)

        with open(file_path, mode='r', encoding='utf-8') as file:
            # 创建一个 CSV 读取器
            csv_reader = csv.reader(file)

            # 读取并打印表头
            headers = next(csv_reader)
            print("Headers:", headers)

            # 打印每一行数据
            for row in csv_reader:
                data_entry = {}
                # obtain data for ['id', 'identifier', 'condition', 'bug', 'fix', 'post_conditional', 'bugType', 'buggy_commit', 'repository_url', 'bug_timestamp', 'num_commits_til_fix']
                id = row[0]
                identifier = row[1]
                condition = row[2]
                bug = row[3]
                fix = row[4]
                post_conditional = row[5]
                bugType = row[6]
                buggy_commit = row[7]
                repository_url = row[8]
                bug_timestamp = row[9]
                num_commits_til_fix = row[10]
                # only use the 10 lines of code before and after the bug
                cut_position = random.randint(10, 20)
                cut_position = 30
                sub_condition = '\n'.join(condition.split('\n')[-cut_position:])
                cut_position = random.randint(10, 20)
                sub_post_conditional = '\n'.join(post_conditional.split('\n')[:cut_position])
                # fix = "Here is a test code!"
                func_src_before = sub_condition + bug + sub_post_conditional
                func_src_after = sub_condition + fix + sub_post_conditional
                changes = get_code_changes(func_src_before, func_src_after)

                data_entry['func_name'] = identifier
                data_entry['func_src_before'] = str(func_src_before)
                data_entry['func_src_after'] = func_src_after
                data_entry['line_changes'] = changes['line_changes']
                data_entry['char_changes'] = changes['char_changes']
                data_entry['commit_link'] = buggy_commit
                data_entry['bug_type'] = bugType
                data_entry['vul_type'] = bugType
                data_entry['repository_url'] = repository_url
                data_entry['sub_condition'] = sub_condition
                data_entry['condition'] = condition
                data_entry['bug'] = bug
                data_entry['fix'] = fix
                data_entries.append(data_entry)


    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


    # shuffle the data_entries
    random.shuffle(data_entries)
    # split the data_entries into train and val
    train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    test_ratio = 0.1
    val_ratio = 0.1
    for ratio in train_ratios:
        # make dirs called ratio/train and ratio/val
        os.makedirs('data/{}/train'.format(ratio), exist_ok=True)
        os.makedirs('data/{}/val'.format(ratio), exist_ok=True)
        os.makedirs('data/{}/test'.format(ratio), exist_ok=True)

        # split the data_entries into train, val, and test
        train_size = int(len(data_entries) * ratio)
        cutoff = int(len(data_entries) * 0.8)
        val_size = int(len(data_entries) * val_ratio)
        test_size = int(len(data_entries) * test_ratio)
        train_data = data_entries[:train_size]

        # so that the test/val data is the same for all ratios
        test_data = data_entries[cutoff: test_size + cutoff]
        val_data = data_entries[test_size + cutoff:]
        
        
        
        # write the data_entries into jsonl files
        for data_entry in train_data:
            bugType = data_entry['bug_type']
            with open('data/{}/train/{}.jsonl'.format(ratio, bugType), 'a') as f:
                json_str = json.dumps(data_entry)
                f.write(str(json_str) + '\n')
        for data_entry in val_data:
            bugType = data_entry['bug_type']
            with open('data/{}/val/{}.jsonl'.format(ratio, bugType), 'a') as f:
                json_str = json.dumps(data_entry)
                f.write(str(json_str) + '\n')
        for data_entry in test_data:
            bugType = data_entry['bug_type']
            with open('data/{}/test/{}.jsonl'.format(ratio, bugType), 'a') as f:
                json_str = json.dumps(data_entry)
                f.write(str(json_str) + '\n')
