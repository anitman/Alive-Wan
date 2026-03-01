#!/bin/bash
# Stage 1: TA-CrossAttn Warmup Training
# Freeze Video DiT and Audio DiT, train only cross-attention layers

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NCCL_DEBUG=INFO

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    training.trainer \
    --config configs/train/stage1_warmup.yaml \
    --output_dir ./outputs/stage1_warmup \
    --resume_from_checkpoint None
