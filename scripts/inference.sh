#!/bin/bash
# Inference script

python -m inference.pipeline \
    --config configs/model/joint_av_dit.yaml \
    --prompt "一只猫在花园里奔跑" \
    --duration 5 \
    --resolution 480p \
    --output ./outputs/demo.mp4 \
    --cfg_scale_text 7.5 \
    --cfg_scale_mutual 2.0
