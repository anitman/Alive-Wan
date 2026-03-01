#!/bin/bash
# Stage 3: Full Model Fine-tuning
# Unfreeze all parameters with very low LR

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NCCL_DEBUG=INFO

python -m torch.distributed.launch \
    --nproc_per_node=16 \
    --nnodes=1 \
    training.trainer \
    --config configs/train/stage3_finetune.yaml \
    --output_dir ./outputs/stage3_finetune \
    --resume_from_checkpoint ./outputs/stage2_joint/checkpoint_last.pt
