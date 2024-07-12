import torch
import argparse
def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", default="meta-llama/Llama-2-7b-hf", type=str)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--total_batch_size", default=8, type=int)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_training_steps", type=int, default=1_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens",  default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default="/data/ajay_data/adaptive_rank_data/downstream")
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--project", type=str, default="adaptive-low-rank")
    parser.add_argument("--name", type=str, default="welore-commonsense-rank50")
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
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument(
        "--dataset", type=str, default="object_tracking", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters", "boolq"], help="dataset used for experiment"
    )
    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")
    parser.add_argument(
        "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument("--do_eval", type=bool, default=True, help="need to run evaluation")
    parser.add_argument("--do_lora", type=bool, default=False, help="need to run evaluation")

    parser.add_argument("--model_rank", type=int, default=50, choices=[0,10,20,30,40,50,60,70])
    parser.add_argument('--min_ratio', type=float, default=0.4999, help='Minimum rank reduction.')
    parser.add_argument("--singular_value_path", default="./data/singular_values_llama-2-7b-base.pt", type=str)
    parser.add_argument(
        "--path_rank_k_checkpoint", default="/data/ajay_data/adaptive_rank_data/finetune_adaptive/welore-single-gpu-rank50/model_checkpoint.pt", type=str
    )
    
    parser.add_argument('--rank_thresold', type=float, default=0.178, help='Rank thresold to prune the model.')
    parser.add_argument("--lora_rank", type=int, default=4, choices=[4, 8, 32])
    parser.add_argument("--lora_alpha", type=int, default=4, choices=[4, 8, 32])
    
    args = parser.parse_args(args)
    return args