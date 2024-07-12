import os
os.environ["WANDB_PROJECT"] = "camera-ready-project"
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from torch.utils.data import Dataset

import GPUtil
from threading import Thread

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import wandb

from tqdm import tqdm
from loguru import logger
from statistics import mean
from functools import partial

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor, QGaLoreAdamW8bit
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import bitsandbytes as bnb

from utils import *
from lib.rank_reduction import do_rank_reduction
from lib.rank_utils import rank_analysis_weight
from lib.eval import eval_ppl
from lib.downstream_utils import *
from lib.downstream_arguments import parse_args

import multiprocessing
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
transformers.logging.set_verbosity_error()

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization(all=True)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


rank_thresold_llama7b_dict = {
    10: 0.065,
    20: 0.084,
    30: 0.115,
    40: 0.145,
    50: 0.175,
    60: 0.215,
    70: 0.26
}
rank_thresold_llama13b_dict = {
    10: 0.0225,
    20: 0.084,
    30: 0.115,
    40: 0.145,
    50: 0.175,
    60: 0.215,
    70: 0.26
}

def load_model(args, rank_k, load_tuned = True):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_config, 
        torch_dtype=torch.bfloat16, 
        cache_dir="/home/aj32632/NeurIPS2024/llm_weights", 
        low_cpu_mem_usage=True, 
        device_map="auto",
        use_auth_token="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"
    )

    
    model.seqlen = 4096
    model.lm_head.weight.requires_grad = False 
    model.model.embed_tokens.weight.requires_grad = False 

    tokenizer = AutoTokenizer.from_pretrained(args.model_config, use_fast=True, padding_side="right", use_auth_token="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")
    tokenizer.pad_token_id = 0


    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if rank_k < 10: return model, tokenizer

    if "7b" in args.model_config:
        args.rank_thresold = rank_thresold_llama7b_dict[args.model_rank]
    elif "13b" in args.model_config:
        args.rank_thresold = rank_thresold_llama13b_dict[args.model_rank]
    else:
        logger.error("Only LLaMa-7b and LLaMa-13b Supported.")
        import sys; sys.exit(0)

    layers_singular_value = torch.load(args.singular_value_path, map_location=torch.device('cpu'))

    rank_pruning = adaptive_rank_pruning(args, args.rank_thresold, layers_singular_value, logger)
    reduced_rank, total_rank = do_rank_reduction(args, model, tokenizer, rank_pruning, args.min_ratio, logger, load_tuned)
    logger.info(f"Effective rank reduction is: {(reduced_rank/total_rank) * 100}")

    model.load_state_dict(torch.load(args.path_rank_k_checkpoint))
    logger.info(f"Model checkpoint loaded successfully.")
    return model, tokenizer


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    global_rank, world_size = 0, 1
    world_size = 1

    if global_rank == 0:
        if not args.unset_wandb:
            wandb.login(key='a5da49ded984c312c13fb590401769bdb3b20099')
            wandb.init(name=args.name)

    
    
    logger.info(f"Global rank {global_rank},  device: {torch.cuda.current_device()}")

    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    ppl = 0
    model, tokenizer = load_model(args, args.model_rank)
    logger.info(f"Loaded model perplexity on C4 : {ppl}")
    
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")

    logger.info("Trainable Parameters are: \n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)

    if not args.unset_wandb:
        wandb.log({"eval_perplexity": ppl,})
        wandb.log({
            "trainable_params_count": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000,
            "total_param_count": sum(p.numel() for p in model.parameters()) / 1_000_000
        })
    
    ########################### Downstream Finetuning ############################
    
    logger.info(f">>>>>>>>>>>>>>>>>>>>>>> Setting up the dataloader for task: {args.dataset} <<<<<<<<<<<<<<<<<<<<<<<<<<")
    
    do_train = True
    if do_train == True:
        dataset = datasets.Dataset.from_dict(dataset_generater(args, eval=False))
        dataset = preprocess_dataset(tokenizer, args.max_length, args.seed, dataset)
        trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args = TrainingArguments(
            per_device_train_batch_size=args.total_batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=10,
            max_steps=args.num_training_steps,
            learning_rate=5e-5,
            bf16=True,
            logging_steps=1,
            output_dir=args.save_dir,
            optim="paged_adamw_8bit",
            save_steps=args.save_every
            ),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
        logger.info("Training started ...")
        import gc; gc.collect()
        torch.cuda.empty_cache()
        monitor = Monitor(50)
        train_result = trainer.train()
        monitor.stop()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        logger.info(metrics) 

    
    if args.do_eval:
        logger.info(f"Evaluation started for task: {args.dataset}")
        dataset = datasets.Dataset.from_dict(dataset_generater(args, eval=args.do_eval))
        correct, total = 0, 0
        dataset = dataset.map(create_prompt_formats_eval)
        for i, item in enumerate(dataset):
            try: 
                prompt = item["text"]
                correct_response = item["raw_y"]
                inputs = tokenizer(prompt, return_tensors="pt")
                outputs = model.generate(input_ids=inputs["input_ids"].cuda(), attention_mask=inputs["attention_mask"].cuda(), max_new_tokens=8, pad_token_id=tokenizer.eos_token_id)
                outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
                predicted_response = parse_predicted(args, prompt, outputs)
                if match_response(args, predicted_response, correct_response) == 0: print(correct_response, predicted_response)
                correct += match_response(args, predicted_response, correct_response)
                total += 1
            except:
                continue
            if i % 100 == 0: 
                logger.info(f"Completed {i}/{len(dataset)} || Current Accuracy: {(correct/(total+1)) * 100:.2f} %")
                print(prompt)
            
        logger.info(f"Accuracy : {(correct/total) * 100:.2f} %")

       
    if not args.unset_wandb:
        wandb.log({"final_accuracy": (correct/total) * 100})
    print("Done")


if __name__ == "__main__":
    print("Starting script")
    args = augument_args(parse_args(None))
    main(args)

    