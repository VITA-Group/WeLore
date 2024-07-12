CUDA_VISIBLE_DEVICES=5 python welore_downstream_finetune.py \
    --name low-rank-downstream-test \
    --path_rank_k_checkpoint /data/ajay_data/adaptive_rank_data/finetune_adaptive/welore-single-gpu-rank50/model_checkpoint.pt \
    --model_rank 50 \
    --dataset strategyqa \
    --total_batch_size 8 \
    --batch_size 8 \
    --num_training_steps 1000