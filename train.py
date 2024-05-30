import logging
import math
import copy
import torch
# import wandb
from datasets import concatenate_datasets
from peft import (
    get_peft_model,
    TaskType,
    LoraConfig,
    IA3Config,
    PromptTuningConfig,
    PrefixTuningConfig
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    default_data_collator,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback, BitsAndBytesConfig
)

from utils import *

logger = logging.getLogger("Hotfix")

class UpdateTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # inputs has two keys: "func_src_before" and "func_src_after"
        # we need to compute the loss for both

        # get the input for buggy code
        inputs_before = {}
        inputs_after = {}
        
        inputs_before["input_ids"] = inputs["input_ids_before"] 
        inputs_before["attention_mask"] = inputs["attention_mask_before"]
        inputs_before["labels"] = inputs["labels_before"]

        # get the input for fixed code
        inputs_after["input_ids"] = inputs["input_ids"]
        inputs_after["attention_mask"] = inputs["attention_mask"]
        inputs_after["labels"] = inputs["labels"]

        
        if return_outputs:
            loss_before, output_before = super().compute_loss(model, inputs_before, return_outputs)
            loss_after, output_after = super().compute_loss(model, inputs_after, return_outputs)
            # for func_src_before, we do gradient ascent
            # for func_src_after, we do gradient descent

            

            return (2*loss_after - loss_before, output_after) if return_outputs else 2*loss_after - loss_before

        else:
            loss_before = super().compute_loss(model, inputs_before, return_outputs)
            loss_after = super().compute_loss(model, inputs_after, return_outputs)

            return 2*loss_after - loss_before


class UnlearnBuggyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # get the input for buggy code
        inputs_buggy = {}
        inputs_buggy["input_ids"] = inputs["input_ids_before"] 
        inputs_buggy["attention_mask"] = inputs["attention_mask_before"]
        inputs_buggy["labels"] = inputs["labels_before"]


        if return_outputs:
            loss, output = super().compute_loss(model, inputs_buggy, return_outputs)
        else:
            loss = super().compute_loss(model, inputs_buggy, return_outputs)

        return (-loss, output) if return_outputs else -loss

class LearnAddedTrainer(Trainer):
    # 在整个修改后的代码上计算loss
    def compute_loss(self, model, inputs, return_outputs=False):
        # change the labels, if not changed, set as -100

        new_inputs = {}
        new_inputs["labels"] = torch.where(inputs["add_weights"] == 0, -100, inputs["labels"])
        new_inputs["input_ids"] = inputs["input_ids"]
        new_inputs["attention_mask"] = inputs["attention_mask"]

        if return_outputs:
            loss, output = super().compute_loss(model, new_inputs, return_outputs)
        else:
            loss = super().compute_loss(model, new_inputs, return_outputs)
        
        return (loss, output) if return_outputs else loss

class LearnAllAfterTrainer(Trainer):
    # 只在添加的部分计算loss
    def compute_loss(self, model, inputs, return_outputs=False):
        # change the labels, if not changed, set as -100

        new_inputs = {}
        new_inputs["labels"] = inputs["labels"]
        new_inputs["input_ids"] = inputs["input_ids"]
        new_inputs["attention_mask"] = inputs["attention_mask"]

        if return_outputs:
            loss, output = super().compute_loss(model, new_inputs, return_outputs)
        else:
            loss = super().compute_loss(model, new_inputs, return_outputs)
        
        return (loss, output) if return_outputs else loss


class UnlearnDeletedTrainer(Trainer):
    # maximize the loss on the deleted part
    def compute_loss(self, model, inputs, return_outputs=False):
        # change the labels, if not changed, set as -100

        new_inputs = {}
        # only calculate loss on the deleted part
        new_inputs["labels"] = torch.where(inputs["delete_weights"] == 0, -100, inputs["labels_before"])
        new_inputs["input_ids"] = inputs["input_ids_before"]
        new_inputs["attention_mask"] = inputs["attention_mask_before"]

        if return_outputs:
            loss, output = super().compute_loss(model, new_inputs, return_outputs)
        else:
            loss = super().compute_loss(model, new_inputs, return_outputs)
        
        # we want to maximize the loss
        return (-loss, output) if return_outputs else -loss

class UnlearnAllBeforeTrainer(Trainer):
    # maximize the loss on the whole original example
    def compute_loss(self, model, inputs, return_outputs=False):
        # change the labels, if not changed, set as -100

        new_inputs = {}
        new_inputs["labels"] = inputs["labels_before"]
        new_inputs["input_ids"] = inputs["input_ids_before"]
        new_inputs["attention_mask"] = inputs["attention_mask_before"]

        if return_outputs:
            loss, output = super().compute_loss(model, new_inputs, return_outputs)
        else:
            loss = super().compute_loss(model, new_inputs, return_outputs)
        
        # we want to maximize the loss
        return (-loss, output) if return_outputs else -loss


class LearnAddedAndUnlearnDeletedTrainer(Trainer):
    # minimize loss on added part (learn added) and maximize loss on deleted part (unlearn deleted)
    def compute_loss(self, model, inputs, return_outputs=False):
        # change the labels, if not changed, set as -100

        new_inputs = {}
        new_inputs["labels"] = torch.where(inputs["delete_weights"] == 0, -100, inputs["labels_before"])
        new_inputs["input_ids"] = inputs["input_ids_before"]
        new_inputs["attention_mask"] = inputs["attention_mask_before"]

        if return_outputs:
            loss, output = super().compute_loss(model, new_inputs, return_outputs)
        else:
            loss = super().compute_loss(model, new_inputs, return_outputs)
        
        # we want to maximize the loss
        return (-loss, output) if return_outputs else -loss


class LearnAddedAndPenalizeDeletedTrainer(Trainer):
    # minimize loss on added part (learn added)
    # penalize the loss on deleted part, but now directly maximize the loss, which may drive the model to produce meaningless code
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs_before = {}
        inputs_after = {}

        inputs_before["input_ids"] = inputs["input_ids_before"] 
        inputs_before["attention_mask"] = inputs["attention_mask_before"]
        inputs_before["labels"] = torch.where(inputs["delete_weights"] == 0, -100, inputs["labels_before"])

        # get the input for fixed code
        inputs_after["input_ids"] = inputs["input_ids"]
        inputs_after["attention_mask"] = inputs["attention_mask"]
        inputs_after["labels"] = torch.where(inputs["add_weights"] == 0, -100, inputs["labels"])

        if return_outputs:
            loss_before, output_before = super().compute_loss(model, inputs_before, return_outputs)
            loss_after, output_after = super().compute_loss(model, inputs_after, return_outputs)

            final_loss = 0.7*loss_after + 0.3*loss_after / (loss_before +  loss_before + 1e-6)
            # if loss_after decreses (i.e., learn added), both parts will decrease
            # if loss_before increases (i.e., penalize deleted), the second term will decrease
            # it means that minimize this final loss will make the model learn added and penalize deleted parts
            return (final_loss, output_after) if return_outputs else 2*loss_after - loss_before
        else:
            loss_before = super().compute_loss(model, inputs_before, return_outputs)
            loss_after = super().compute_loss(model, inputs_after, return_outputs)

            final_loss = 0.7*loss_after + 0.3*loss_after / (loss_before +  loss_before + 1e-6)

            return final_loss

def enhance_correctness_class(base_class, args_info):
    
    class_name = base_class.__name__ + "EnhancedCorrectness"
    print(base_class.__name__)
    def __init__(self, *args, **kwargs):
        # Initialize the base class
        super(base_class, self).__init__(*args, **kwargs)
        # load the original model using args.model_name_or_path
        self.model_original = AutoModelForCausalLM.from_pretrained(args_info.model_name_or_path)
        self.model_original.to(args_info.device)

    def compute_loss(self, model, inputs, return_outputs=False):     
        # get the loss from the base class
        if return_outputs:
            loss, output = base_class.compute_loss(self, model=model, inputs=inputs, return_outputs=return_outputs)
        else:
            loss =  base_class.compute_loss(self, model=model, inputs=inputs, return_outputs=return_outputs)
            
        
        # add the correctness loss (KL divergence) to the original loss
        # compute the KL divergence between (1) the new model's probability distribution and (2) the original model's probability distribution
        new_input = {}
        new_input["input_ids"] = inputs["input_ids"]
        new_input["attention_mask"] = inputs["attention_mask"]
        new_input["labels"] = inputs["labels"]
        
        # get the logits for the new model
        model.eval()  # change the model to eval mode
        with torch.no_grad():
            outputs = model(**new_input)
            logits_new = outputs.logits
            # get the probability distribution
            probability_new = torch.nn.functional.softmax(logits_new, dim=-1)
        model.train() # change the model back to train mode
        
        
        # get the logits for the original model
        self.model_original.eval()  # change the model to eval mode
        with torch.no_grad():
            outputs = self.model_original(**new_input)
            logits_original = outputs.logits
            # get the probability distribution
            probability_original = torch.nn.functional.softmax(logits_original, dim=-1)
            
    
        
        # compute the KL divergence
        loss_fct = torch.nn.KLDivLoss(log_target=True, reduction='none')
        kl_loss = loss_fct(probability_new,
                probability_original)
        
        kl_loss = kl_loss.sum(dim=1).mean()
        
        if return_outputs:
            return (0.7*loss + 0.3*kl_loss, output)
        else:
            return 0.7*loss + 0.3*kl_loss
        
    
    class_attributes = {
        "compute_loss": compute_loss,
        "__init__": __init__
    } # add the new attributes to the class
    
    # create a new class with the new attributes
    new_class = type(class_name, (base_class,), class_attributes)

    return new_class

class SaveBestModelCallback(TrainerCallback):
    def __init__(self, trainer, eval_steps):
        self.trainer = trainer
        self.eval_steps = eval_steps
        self.best_loss = float("inf")
        self.saved_models_dir = []

    def on_step_end(self, args, state, control, model, tokenizer, **kwargs):
        if state.global_step % self.eval_steps == 0:
            evaluation_results = self.trainer.evaluate()
            eval_loss = evaluation_results["eval_loss"]

            if eval_loss < self.best_loss:
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                self.best_loss = eval_loss

                if "wandb" in args.report_to:
                    wandb.run.summary["best_evaluation_loss"] = eval_loss


def load_model_and_tokenizer(args):
    model_cls = AutoModelForSeq2SeqLM if "codet5" in args.model_name_or_path else AutoModelForCausalLM
    task_type = TaskType.SEQ_2_SEQ_LM if "codet5" in args.model_name_or_path else TaskType.CAUSAL_LM

    model_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": True
    }
    if args.tuning_method != "ft":
        model_kwargs["torch_dtype"] = torch.float16

    if args.tuning_method == "qlora-8bit":
        qconfig = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["quantization_config"] = qconfig
    elif args.tuning_method == "qlora-4bit":
        qconfig = BitsAndBytesConfig(load_in_4bit=True,
                                     bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True,
                                     bnb_4bit_compute_dtype=torch.float16)
        model_kwargs["quantization_config"] = qconfig

    model = model_cls.from_pretrained(args.model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if args.tuning_method in ["lora", "qlora-8bit", "qlora-4bit"]:
        peft_config = LoraConfig(task_type=task_type,
                                 r=args.lora_r,
                                 lora_alpha=args.lora_alpha,
                                 target_modules=LORA_IA3_TARGET_MODULES[args.model_name]["target_modules_lora"],
                                 lora_dropout=args.lora_dropout,
                                 bias="none")
    elif args.tuning_method == "ia3":
        peft_config = IA3Config(task_type=task_type,
                                target_modules=LORA_IA3_TARGET_MODULES[args.model_name]["target_modules_ia3"],
                                feedforward_modules=LORA_IA3_TARGET_MODULES[args.model_name]["ff_modules"])
    elif args.tuning_method == "prompt-tuning":
        peft_config = PromptTuningConfig(task_type=task_type,
                                         num_virtual_tokens=args.prompt_num_virtual_tokens,
                                         prompt_tuning_init="TEXT",
                                         prompt_tuning_init_text="Generate Python code given a natural language "
                                                                 "instruction.",
                                         tokenizer_name_or_path=args.model_name_or_path)
    elif args.tuning_method == "prefix-tuning":
        peft_config = PrefixTuningConfig(task_type=task_type,
                                         num_virtual_tokens=args.prefix_num_virtual_tokens)

    if args.tuning_method != "ft":
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    if "codet5" not in args.model_name_or_path:
        tokenizer.padding_side = "left"

    return model, tokenizer


def run_train_hotfix(args):
    # load datasets
    dataset = load_sstubs_train_dataset()
    dataset_val = load_sstubs_valid_dataset()
    
    # dataset = dataset.select(range(10)) # REMEBER TO REMOVE THIS, only for testing purposes
    intent_column = "sub_condition" # prompt to show intent
    code_column = "fix" # the python commands that serve as the target

    # load models
    global tokenizer 
    model, tokenizer = load_model_and_tokenizer(args)

    def preprocess_function(example):
        """
        TODO: document this function
        """
        # KEYS in example: line_changes, char_changes, func_src_after, func_src_before

        data_to_return = {}
        code_after = example["func_src_after"]
        tokenized_code_after = tokenizer.encode_plus(
            code_after,
            max_length=512,
            truncation=True,
            add_special_tokens=False,
            padding="max_length")
        tokens_after = tokenized_code_after['input_ids'] # input_ids

        # if len(tokens_after) > 512:
        #     return None

        add_weights = [0] * len(tokens_after)
        # 没修改的地方设置0，修改的地方设置1
        # 现在找到修改的地方
        # this is after, we need to find added parts
        for change in example["char_changes"]["added"]:
            try:
                char_start = change["char_start"]
                char_start_idx = tokenized_code_after.char_to_token(char_start)
                char_end = change["char_end"]
                char_end_idx = tokenized_code_after.char_to_token(char_end) - 1

                # now we find the tokens that correspond to the added characters
                for char_idx in range(char_start_idx, char_end_idx+1):
                    # 这个range就是修改的位置
                    add_weights[char_idx] = 1 # 将每个修改的位置的权重设置为1
            except:
                pass

        data_to_return["input_ids"] = tokenized_code_after.data["input_ids"]
        data_to_return["attention_mask"] = tokenized_code_after["attention_mask"]
        data_to_return["add_weights"] = add_weights
        data_to_return["labels"] = tokenized_code_after["input_ids"]

        code_before = example["func_src_before"]
        tokenized_code_before = tokenizer.encode_plus(
            code_before,
            max_length=512,
            truncation=True,
            add_special_tokens=False,
            padding="max_length")

        tokens_before = tokenized_code_before['input_ids']
        # if len(tokens_before) > 512:
        #     return None

        delete_weights = [0] * len(tokens_before)
        # 没修改的地方设置0，修改的地方设置1
        # 现在找到修改的地方
        # this is after, we need to find deleted parts
        for change in example["char_changes"]["deleted"]:
            try:
                char_start = change["char_start"]
                char_start_idx = tokenized_code_before.char_to_token(char_start)
                char_end = change["char_end"]
                char_end_idx = tokenized_code_before.char_to_token(char_end) - 1

                # now we find the tokens that correspond to the added characters
                for char_idx in range(char_start_idx, char_end_idx+1):
                    # 这个range就是修改的位置
                    delete_weights[char_idx] = 1
            except:
                # this change is not in the tokenized part
                pass
        
        data_to_return["delete_weights"] = delete_weights

        data_to_return["input_ids_before"] = tokenized_code_before["input_ids"] 
        data_to_return["attention_mask_before"] = tokenized_code_before["attention_mask"] 
        data_to_return["labels_before"] = tokenized_code_before["input_ids"]

        return data_to_return

    def preprocess_function_buggy_fixed(example):
        
        data_type = "func_src_before"

        # tokenize the target
        model_inputs = tokenizer(example[data_type],
                                    max_length=512 - 1,
                                    truncation=True,
                                    add_special_tokens=False,
                                    padding="max_length")



        model_inputs["input_ids"] = model_inputs["input_ids"] + [tokenizer.eos_token_id]
        model_inputs["attention_mask"] = model_inputs["attention_mask"] + [1]
        model_inputs["labels"] = model_inputs["input_ids"]


        return model_inputs


    def preprocess_function_seq2seq(example):
        prompt = "Generate Python code: ### Instruction:\n" + example[intent_column] + "\n### Response:\n"
        model_inputs = tokenizer(prompt, max_length=args.max_input_length, padding="max_length", truncation=True)
        labels = tokenizer(example[code_column], max_length=args.max_target_length, padding="max_length", truncation=True)

        labels["input_ids"] = [l if l != tokenizer.pad_token_id else -100 for l in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    intent_column = "nl" # prompt to show intent
    code_column = "cmd" # the python commands that serve as the target


    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)


    # tokenize_fn = preprocess_function_seq2seq if "codet5" in args.model_name_or_path else preprocess_function

    tokenize_fn = preprocess_function
    columns_to_remove = ['func_name', 'func_src_before', 'func_src_after', 'line_changes', 'char_changes', 'commit_link', 'bug_type', 'vul_type', 'repository_url', 'sub_condition', 'condition', 'bug', 'fix']
    dataset = dataset.map(tokenize_fn,
                            num_proc=args.num_workers,
                            desc="Generating samples features.")   

    dataset = dataset.remove_columns(columns_to_remove)
    dataset_val = dataset_val.map(tokenize_fn,
                            num_proc=args.num_workers,
                            desc="Generating samples features.")   
    dataset_val = dataset_val.remove_columns(columns_to_remove)

    # print(dataset[0])
    # print(dataset[0].keys())
    # exit()
    # if args.loss_mode == "learn_fix":
    #     dataset = data_buggy_and_fixed["func_src_after"] # to not break the original flow
    # elif args.loss_mode == "unlearn_buggy":
    #     dataset = data_buggy_and_fixed["func_src_before"]
    # elif args.loss_mode == "update":
    #     dataset = data_buggy_and_fixed

    # dataset = dataset.map(tokenize_fn,
    #                       num_proc=args.num_workers,
    #                       remove_columns=dataset.column_names,
    #                       desc="Generating samples features.")


    dataset = dataset.shuffle(seed=42)

    
    n_samples = len(dataset)
    n_samples_per_step = args.batch_size * args.num_gpus * args.gradient_accumulation_steps
    eval_steps = math.ceil((n_samples // n_samples_per_step) * args.ratio_samples_per_eval_step)

    training_args_cls = Seq2SeqTrainingArguments if "codet5" in args.model_name_or_path else TrainingArguments
    training_args = training_args_cls(
        output_dir=args.run_dir,
        evaluation_strategy="no",
        save_strategy="no",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.05,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        optim="adafactor",
        logging_strategy="steps",
        logging_steps=10,
        fp16=True,
        remove_unused_columns=False,
        report_to=["wandb"] if args.use_wandb else ["none"]
    )

    # TODO: may put them into a dictionary
    if args.loss_mode == "learn_fix":
        trainer_cls = Seq2SeqTrainer if "codet5" in args.model_name_or_path else Trainer
    elif args.loss_mode == "unlearn_buggy":
        trainer_cls = UnlearnBuggyTrainer
    elif args.loss_mode == "update":
        trainer_cls = UpdateTrainer
    elif args.loss_mode == "learn_added":
        trainer_cls = LearnAddedTrainer
    elif args.loss_mode == "all_after":
        trainer_cls = LearnAllAfterTrainer
    elif args.loss_mode == "unlearn_deleted":
        trainer_cls = UnlearnDeletedTrainer
    elif args.loss_mode == "unlearn_all_before":
        trainer_cls = UnlearnAllBeforeTrainer
    elif args.loss_mode == "learn_added_unlearn_deleted":
        trainer_cls = LearnAddedAndUnlearnDeletedTrainer
    elif args.loss_mode == "learn_added_penalize_deleted":
        trainer_cls = LearnAddedAndPenalizeDeletedTrainer
    else:
        raise ValueError(f"Invalid loss mode: {args.loss_mode}")

    if args.enhance_correctness is True:
        print("Enhancing correctness")
        trainer_cls = enhance_correctness_class(trainer_cls, args)
    

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset_val,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.add_callback(SaveBestModelCallback(trainer, eval_steps))
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation loss before training: {round(eval_results['eval_loss'], 4)}")
    trainer.train()

    exit()



def run_train(args):
    if args.dataset == "joint":
        dataset = load_conala_train_dataset()
        codealpaca = load_codealpaca_train_dataset()
        dataset["train"] = concatenate_datasets([dataset["train"], codealpaca["train"]])
        dataset["validation"] = concatenate_datasets([dataset["validation"], codealpaca["validation"]])
        dataset.shuffle(args.seed)
    else:
        dataset_loading_func = globals().get(f"load_{args.dataset}_train_dataset")
        dataset = dataset_loading_func()

    intent_column = "nl" # prompt to show intent
    code_column = "cmd" # the python commands that serve as the target
    print(dataset["train"], dataset["validation"])

    for example in dataset["train"].select(range(3)):
        print(f"### Instruction:\n{example[intent_column]}\n### Response:\n{example[code_column]}")
        print("=" * 100)

    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)

    def preprocess_function(example):
        """
        # we tokenize, pad and truncate the samples in the following way:
        #   <pad><pad>...### Instruction:\n<intent>\n### Answer:\n<snippet><eos>
        #
        #   - prompt tokens `<pad><pad>...<intent + \n>` are ignored in the computation of the loss (-100 labels)
        #   - `<eos>` delimits the snippet and allows the model to have more focused predictions at inference
        """
        tokenized_target = tokenizer(example[code_column],
                                     max_length=args.max_target_length - 1,
                                     truncation=True,
                                     # incoder adds eos token before the start of a sequence -> ignore
                                     add_special_tokens=False)
        tokenized_target["input_ids"] = tokenized_target["input_ids"] + [tokenizer.eos_token_id]
        tokenized_target["attention_mask"] = tokenized_target["attention_mask"] + [1]

        prompt = "### Instruction:\n" + example[intent_column] + "\n### Response:\n"
        max_prompt_len = (args.max_input_length + args.max_target_length) - \
                         len(tokenized_target["input_ids"])
        model_inputs = tokenizer(prompt, max_length=max_prompt_len,  padding="max_length", truncation=True)

        model_inputs["labels"] = [-100] * len(model_inputs["input_ids"]) + tokenized_target["input_ids"]
        model_inputs["input_ids"] = model_inputs["input_ids"] + tokenized_target["input_ids"]
        model_inputs["attention_mask"] = model_inputs["attention_mask"] + tokenized_target["attention_mask"]

        return model_inputs

    def preprocess_function_seq2seq(example):
        prompt = "Generate Python code: ### Instruction:\n" + example[intent_column] + "\n### Response:\n"
        model_inputs = tokenizer(prompt, max_length=args.max_input_length, padding="max_length", truncation=True)
        labels = tokenizer(example[code_column], max_length=args.max_target_length, padding="max_length", truncation=True)

        labels["input_ids"] = [l if l != tokenizer.pad_token_id else -100 for l in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenize_fn = preprocess_function_seq2seq if "codet5" in args.model_name_or_path else preprocess_function
    dataset = dataset.map(tokenize_fn,
                          num_proc=args.num_workers,
                          remove_columns=dataset["train"].column_names,
                          desc="Generating samples features.")

    # exit()

    n_samples = len(dataset["train"])
    n_samples_per_step = args.batch_size * args.num_gpus * args.gradient_accumulation_steps
    eval_steps = math.ceil((n_samples // n_samples_per_step) * args.ratio_samples_per_eval_step)

    training_args_cls = Seq2SeqTrainingArguments if "codet5" in args.model_name_or_path else TrainingArguments
    training_args = training_args_cls(
        output_dir=args.run_dir,
        evaluation_strategy="no",
        save_strategy="no",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.05,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        optim="adafactor",
        logging_strategy="steps",
        logging_steps=10,
        fp16=True,
        report_to=["wandb"] if args.use_wandb else ["none"]
    )
    trainer_cls = Seq2SeqTrainer if "codet5" in args.model_name_or_path else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        # eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.add_callback(SaveBestModelCallback(trainer, eval_steps))
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation loss before training: {round(eval_results['eval_loss'], 4)}")
    trainer.train()