import os

import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor, QGaLoreAdamW8bit
import bitsandbytes as bnb

from utils import *
from lib.rank_reduction import do_rank_reduction
from lib.rank_utils import rank_analysis_weight
from lib.eval import eval_ppl

transformers.logging.set_verbosity_error()

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

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", default="meta-llama/Llama-2-7b-hf", type=str)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--total_batch_size", default=4, type=int)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--lr", type=float, default=5e-05)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=str, default="camera-ready-project")
    parser.add_argument("--name", type=str, default="low-rank-continual-finetune-test")
    parser.add_argument("--unset_wandb", action="store_true")
    parser.add_argument("--unset_save", action="store_true")
    parser.add_argument("--grad_clipping", type=float, default=0.0)  
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")
    parser.add_argument("--proj_bits", type=int, default=8)
    parser.add_argument("--proj_quant", action='store_true')
    parser.add_argument("--proj_group_size", type=int, default=128)
    parser.add_argument("--single_gpu", default=False, action="store_true")
    parser.add_argument("--singular_value_path", default="./data/singular_values_llama-2-7b-base.pt", type=str)
    parser.add_argument('--model_rank', type=int, default=50, help='Required Pruning Rank.')
    parser.add_argument('--min_ratio', type=float, default=0.4999, help='Minimum rank reduction.')
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--unset_welore_pos", action="store_true")
    
    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args



@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    _time = time.time()
    val_data = datasets.load_dataset("c4", "en", split="validation", streaming=True) #DGX
    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")


    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if "7b" in args.model_config:
        args.rank_thresold = rank_thresold_llama7b_dict[args.model_rank]
    elif "13b" in args.model_config:
        args.rank_thresold = rank_thresold_llama13b_dict[args.model_rank]
    else:
        logger.error("Only LLaMa-7b and LLaMa-13b Supported.")
        import sys; sys.exit(0)


    global_rank, world_size = 0, 1
    world_size = 1
    logger.info(f"Global rank {global_rank},  device: {torch.cuda.current_device()}")

    
    if global_rank == 0:
        if not args.unset_wandb:
            wandb.init(project=args.project, name=args.name)
            
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    tokenizer = AutoTokenizer.from_pretrained(args.model_config, use_fast=True, padding_side="right", use_auth_token="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")
    tokenizer.pad_token_id = 0

    

    ################# Model Loading and Pruning ###################
    model = AutoModelForCausalLM.from_pretrained(
        args.model_config, 
        torch_dtype=torch.bfloat16, 
        # cache_dir="../llm_weights", 
        cache_dir="/data/ajay_data/llama2_models",
        low_cpu_mem_usage=True, 
        device_map="auto",
        use_auth_token="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"
    )

    model.seqlen = 4096
    model.lm_head.weight.requires_grad = False 
    model.model.embed_tokens.weight.requires_grad = False 

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    
    if os.path.exists(args.singular_value_path):
        layers_singular_value = torch.load(args.singular_value_path, map_location=torch.device('cpu'))
    else:
        layers_singular_value = rank_analysis_weight(args, model, tokenizer, None)
        torch.save(layers_singular_value, args.singular_value_path)

    layers_singular_value = torch.load(args.singular_value_path, map_location=torch.device('cpu'))
    logger.info("------------------- Dense Model Loaded --------------------")
    
    dense_ppl, reduction_ppl = 0, 0
    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Total params BEFORE Rank Reduction: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params BEFORE Rank Reduction: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")

    dense_ppl = eval_ppl(model, tokenizer, None, "c4")
    logger.info(f"Before Rank Reduction PPL on C4: {dense_ppl}\n")

    rank_pruning = adaptive_rank_pruning(args, args.rank_thresold, layers_singular_value, logger)
    reduced_rank, total_rank = do_rank_reduction(args, model, tokenizer, rank_pruning, args.min_ratio, logger, False)
    logger.info(f"\n{model}\n")
    logger.info(f"****** Pruning completed with Rank Reduced/Total Rank : {reduced_rank}/{total_rank} ({(reduced_rank/total_rank)*100:.3f} %) ******")
    
    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Total params AFTER Rank Reduction: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params AFTER Rank Reduction: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")

    reduction_ppl = eval_ppl(model, tokenizer, None, "c4")
    logger.info(f"After Rank Reduction PPL on C4: {reduction_ppl}\n")

    logger.info("Trainable Parameters are: \n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)

    
    logger.info("------------------- Finetuning Log --------------------")
    if not args.unset_save:
        current_model_directory = f"{args.save_dir}/{args.name}"
        os.makedirs(current_model_directory, exist_ok=True)
        torch.save(model.state_dict(), f"{current_model_directory}/model_checkpoint.pt")
    
    seed_for_shuffle = 42 
    logger.info(f"Loading and Shuffling data with seed {seed_for_shuffle} ...")
    data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0
    
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
    })
    pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    if not args.unset_wandb:
        wandb.log({"eval_perplexity": reduction_ppl,}, step=global_step)
        wandb.log({
            "rank_reduction": (reduced_rank/total_rank)*100,
            "trainable_params_count": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000,
            "total_param_count": sum(p.numel() for p in model.parameters()) / 1_000_000
        })

    layer_wise_flag = False
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.beta1)


    if not layer_wise_flag:
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
    
    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################
    import gc; gc.collect()
    torch.cuda.empty_cache()
    for batch_idx, batch in enumerate(dataloader):
        global_step += 1
        local_step += 1

        if update_step > args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            break

        batch = {k: v.cuda() for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model(**batch, labels=labels).loss
        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        # add grad clipping
        if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        if global_rank == 0: pbar.update(1)
        
        if not layer_wise_flag:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time

    ###############################################################
        #Saving model checkpoints
        if update_step % args.save_every == 0:
            if not args.unset_save:
                current_model_directory = f"{args.save_dir}/{args.name}"
                logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
                os.makedirs(current_model_directory, exist_ok=True)

                optimizer_checkpoint = {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "update_step": update_step,
                    "global_step": global_step,
                    "config": run_config,
                    "wandb": wandb.run.dir if not args.unset_wandb else None,
                    "dtype": args.dtype,
                }
                training_state_checkpoint = {
                    "global_step": global_step,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "tokens_seen_before": tokens_seen_before,
                    "update_time": update_time,
                }
                with open(f"{current_model_directory}/training_state.json", "w") as f:
                    json.dump(training_state_checkpoint, f, indent=4)
                torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")
                torch.save(model.state_dict(), f"{current_model_directory}/model_checkpoint.pt")
            

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            reduction_ppl = eval_ppl(model, tokenizer, None, "c4")
            logger.info(f"Perpelxity at {update_step} step: {reduction_ppl}\n")
            if global_rank == 0:
                if not args.unset_wandb:
                    wandb.log({
                        "eval_perplexity": reduction_ppl,
                        },
                        step=global_step,
                    )
        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0:
            if not args.unset_wandb:
                wandb.log({
                    "loss": loss.item(),
                    "lr": lr,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "throughput_tokens": tokens_in_update / update_time,
                    "throughput_examples": args.total_batch_size / update_time,
                    "throughput_batches": batches_in_update / update_time,
                    },
                    step=global_step,
                )
        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()
    import gc; gc.collect()
    torch.cuda.empty_cache()
    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)