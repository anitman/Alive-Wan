#!/bin/bash
# Stage 2: Joint AV DiT Fine-tuning
# Unfreeze Audio DiT with lower LR, keep Video DiT frozen

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NCCL_DEBUG=INFO

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    training.trainer \
    --config configs/train/stage2_joint.yaml \
    --output_dir ./outputs/stage2_joint \
    --resume_from_checkpoint ./outputs/stage1_warmup/checkpoint_last.pt
