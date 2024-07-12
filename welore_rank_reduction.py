import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

import torch
import argparse
import numpy as np

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from importlib.metadata import version

from timeit import default_timer as timer
from datetime import timedelta
from loguru import logger
import wandb

from utils import *
from lib.rank_reduction import do_rank_reduction
from lib.rank_utils import rank_analysis_weight

from lib.eval import eval_ppl
transformers.logging.set_verbosity_error()

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto",
        use_auth_token="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"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

rank_thresold_llama7b_dict = {
    10: 0.0225,
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="meta-llama/Llama-2-7b-hf", type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument("--cache_dir", default="./llama_weights", type=str )
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--estimate_rank', type=bool, default=True, help='Check if the layerwise singular values need to be calculated.')
    
    parser.add_argument('--model_rank', type=int, default=10, help='Required Pruning Rank.')
    
    parser.add_argument("--data_dir", "-d", type=str, default="./data")
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--project", default="welore-project", type=str)
    parser.add_argument("--name", default="low-rank-pruning-test", type=str)
    parser.add_argument("--singular_value_path", default="./data/singular_values_llama-2-7b.pt", type=str)
    parser.add_argument('--min_ratio', type=float, default=0.0, help='Minimum rank reduction.')


    args = parser.parse_args()
    if "7b" in args.model:
        args.rank_thresold = rank_thresold_llama7b_dict[args.model_rank]
    elif "13b" in args.model:
        args.rank_thresold = rank_thresold_llama13b_dict[args.model_rank]
    else:
        logger.error("Only LLaMa-7b and LLaMa-13b Supported.")
        import sys; sys.exit(0)

    wandb.init(project=args.project, name=args.name)


    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)
    
 
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model_name = args.model.split("/")[-1]
    logger.info(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)

    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")

    logger.info(f"{args}")

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token="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")

    logger.info("Model and tokenizer loaded")

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    logger.info("use device ", device)

    layers_singular_value = None
    if args.estimate_rank:
        if os.path.exists(args.singular_value_path):
            layers_singular_value = torch.load(args.singular_value_path, map_location=torch.device('cpu'))
        else:
            layers_singular_value = rank_analysis_weight(args, model, tokenizer, device)
            torch.save(layers_singular_value, args.singular_value_path)

    print ("device",device)

    ############################################# dense ############################################

    dense_ppl = eval_ppl(model, tokenizer, None, "c4")
    logger.info(f"Before Rank Reduction PPL on C4: {dense_ppl}\n")

    ############################################# svd ############################################
    

    rank_pruning = adaptive_rank_pruning(args, args.rank_thresold, layers_singular_value, logger)
    reduced_rank, total_rank = do_rank_reduction(args, model, tokenizer, rank_pruning, args.min_ratio, logger, False)
    

    reduction_svd_ppl = eval_ppl(model, tokenizer, None, "c4")
    logger.info(f"After Rank Reduction PPL on C4: {reduction_svd_ppl}\n")
    
    wandb.log({"Dense_PPL": dense_ppl})
    wandb.log({"Compressed_PPL": reduction_svd_ppl})
    wandb.log({"Reduction_Ratio": (reduced_rank/total_rank)*100})

if __name__ == '__main__':
    main()