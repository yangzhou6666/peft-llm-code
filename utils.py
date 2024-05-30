import json

from datasets import load_dataset

LORA_IA3_TARGET_MODULES = {
    "codegen-350M-mono": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codegen-350M-multi": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codegen-350M-nl": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codegen-2B-mono": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codegen-2B-multi": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codegen-2B-nl": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codegen-6B-mono": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codegen-6B-multi": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codegen-6B-nl": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codet5p-220m": {
        "target_modules_lora": ["q", "v", "k"],
        "target_modules_ia3": ["q", "v", "k", "wi", "wo"],
        "ff_modules": ["wi", "wo"]
    },
    "codet5p-770m": {
        "target_modules_lora": ["q", "v", "k"],
        "target_modules_ia3": ["q", "v", "k", "wi", "wo"],
        "ff_modules": ["wi", "wo"]
    },
    "codet5p-2b": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codet5p-6b": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codegen2-1B": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codegen2-3_7B": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codegen2-7B": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "CodeLlama-7b-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    },
    "CodeLlama-7b-Instruct-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    },
    "CodeLlama-7b-Python-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    },
    "CodeLlama-13b-Python-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    },
    "CodeLlama-34b-Python-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    }
}


def trim_code(completion, stop_tokens):
    for stop_token in stop_tokens:
        if stop_token in completion:
            completion = completion[:completion.find(stop_token)]
    return completion


def load_conala_train_dataset():
    datasets = load_dataset("neulab/docprompting-conala")
    datasets = datasets.filter(lambda x: x["nl"] is not None)
    datasets = datasets.filter(lambda x: x["cmd"] is not None)
    del datasets["test"]
    return datasets


def load_conala_test_dataset():
    dataset = load_dataset("neulab/docprompting-conala")["test"]
    return dataset


def load_sstubs_train_dataset():
    paths = [
        "sstubs_data/data/0.8/train/CHANGE_CALLER_IN_FUNCTION_CALL.jsonl",
        "sstubs_data/data/0.8/train/LESS_SPECIFIC_IF.jsonl",
        "sstubs_data/data/0.8/train/CHANGE_IDENTIFIER.jsonl",
        "sstubs_data/data/0.8/train/CHANGE_NUMERAL.jsonl",
        "sstubs_data/data/0.8/train/CHANGE_OPERAND.jsonl",
        "sstubs_data/data/0.8/train/CHANGE_OPERATOR.jsonl",
        "sstubs_data/data/0.8/train/CHANGE_UNARY_OPERATOR.jsonl",
        "sstubs_data/data/0.8/train/DIFFERENT_METHOD_SAME_ARGS.jsonl",
        "sstubs_data/data/0.8/train/MORE_SPECIFIC_IF.jsonl",
        "sstubs_data/data/0.8/train/OVERLOAD_METHOD_DELETED_ARGS.jsonl",
        "sstubs_data/data/0.8/train/OVERLOAD_METHOD_MORE_ARGS.jsonl",
        "sstubs_data/data/0.8/train/SWAP_BOOLEAN_LITERAL.jsonl",
    ]
    dataset = load_dataset("json", data_files=paths, split="train")
    return dataset

def load_sstubs_valid_dataset():
    paths = [
        "sstubs_data/data/0.8/val/CHANGE_CALLER_IN_FUNCTION_CALL.jsonl",
        "sstubs_data/data/0.8/val/LESS_SPECIFIC_IF.jsonl",
        "sstubs_data/data/0.8/val/CHANGE_IDENTIFIER.jsonl",
        "sstubs_data/data/0.8/val/CHANGE_NUMERAL.jsonl",
        "sstubs_data/data/0.8/val/CHANGE_OPERAND.jsonl",
        "sstubs_data/data/0.8/val/CHANGE_OPERATOR.jsonl",
        "sstubs_data/data/0.8/val/CHANGE_UNARY_OPERATOR.jsonl",
        "sstubs_data/data/0.8/val/DIFFERENT_METHOD_SAME_ARGS.jsonl",
        "sstubs_data/data/0.8/val/MORE_SPECIFIC_IF.jsonl",
        "sstubs_data/data/0.8/val/OVERLOAD_METHOD_DELETED_ARGS.jsonl",
        "sstubs_data/data/0.8/val/OVERLOAD_METHOD_MORE_ARGS.jsonl",
        "sstubs_data/data/0.8/val/SWAP_BOOLEAN_LITERAL.jsonl",
    ]
    dataset = load_dataset("json", data_files=paths, split="train")
    return dataset

def load_sstubs_test_dataset():
    pass

def load_privacy_train_dataset():
    pass

def load_privacy_test_dataset():
    pass


def load_codealpaca_train_dataset():
    dataset = load_dataset("antolin/codealpaca-filtered")
    dataset["validation"] = dataset["valid"]
    del dataset["test"], dataset["valid"]
    return dataset


def load_codealpaca_test_dataset():
    return load_dataset("antolin/codealpaca-filtered")["test"]


def load_conala_icl_examples():
    with open("conala_icl_examples.json") as f:
        examples = json.load(f)
    return examples


def load_codealpaca_icl_examples():
    with open("codealpaca_icl_examples.json") as f:
        examples = json.load(f)
    return examples


def load_odex_test_dataset():
    dataset = load_dataset("neulab/odex")["test"]
    conala = load_dataset("neulab/docprompting-conala")["train"]

    # make sure we remove test samples that appear in the fine-tuning dataset to avoid data leakage
    dataset = dataset.filter(lambda example: example["intent"] not in conala["nl"])

    return dataset


if __name__ == '__main__':
    # only for testing purposes
    pass