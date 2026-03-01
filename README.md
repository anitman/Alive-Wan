# Alive-Wan2X

<div align="center">

**Audio-Visual Video Generation Built on Wan2.x**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org)

**语言 / Language:**

<details open>
<summary>🇨🇳 中文</summary>

---

基于 Wan2.1/Wan2.2 重构的 Alive 有声视频生成模型，支持文本/图像到带声音视频的生成。

## 功能特性

- **T2VA**（Text-to-Video&Audio）：从文本生成带声音的视频
- **I2VA**（Image-to-Video&Audio）：从图像生成带声音的视频
- **R2VA**（Reference-to-Video&Audio）：基于参考人物形象的有声视频动画
- 支持 480P 基础生成 + 1080P 级联精炼
- 5~10 秒任意时长生成
- 多条件 CFG 控制音视频同步强度

## 架构概述

基于 **Dual Stream + Single Stream** 范式：

```
输入层
├── Video Latents (Wan-VAE)
├── Audio Latents (WavVAE)
└── Text Embeddings (Qwen3-8B/4B + Qwen2.5-32B optional)

Dual Stream 阶段（M=16 blocks）
├── Video DiT Blocks  [B, T_v, 4096]
├── Audio DiT Blocks  [B, T_a, 1536]
└── TA-CrossAttn (音视频跨模态注意力)

Single Stream 阶段（N=40 blocks）
└── 联合音视频处理 [B, T_v+T_a, 4096]
```

### 核心技术

- **UniTemp-RoPE**：统一时间位置编码，实现音视频物理时间对齐（视频 1/6s/token，音频 1/50s/token）
- **TA-CrossAttn**：时间对齐交叉注意力，确保音视频同步
- **双条件控制**：Speech Text + Descriptive Prompt 双路控制
- **Multi-CFG**：`v_guided = v_uncond + cfg_text*(v_text - v_uncond) + cfg_mutual*(v_cond - v_text)`
- **级联精炼**：480P → 1080P 高分辨率增强

## 快速开始

### 环境要求

```
Python >= 3.10
CUDA >= 11.7
PyTorch >= 2.1
```

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/alive-wan2x.git
cd alive-wan2x

# 安装依赖
pip install -r requirements.txt
pip install -e .

# 下载模型权重
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./models/wan2.1-t2v-14b
huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/qwen3-8b
# 或使用更轻量的 4B 版本：
# huggingface-cli download Qwen/Qwen3-4B --local-dir ./models/qwen3-4b
```

### 推理

```bash
# 文本生成带声音视频
python -m inference.pipeline \
    --prompt "一只猫在花园里奔跑" \
    --duration 5 \
    --resolution 480p \
    --output output.mp4

# 或使用命令行入口
alive-infer --prompt "prompt text" --output output.mp4
```

## 训练

### 数据准备

数据集支持 JSON/JSONL 格式，每条样本包含 `video_path`、`audio_path`、`caption` 字段。推荐数据集：VGGSound、AudioSet、LibriTTS、AudioCaps。

### 多阶段训练策略

| 阶段 | 冻结参数 | 可训练参数 | 学习率 |
|------|---------|-----------|-------|
| Stage 1: TA-CrossAttn 暖启动 | Video DiT + Audio DiT | CrossAttn only | 1e-4 |
| Stage 2: 联合微调 | Video DiT + Text | Audio DiT + CrossAttn | 1e-5 |
| Stage 3: 全模型微调 | Text encoder | All params | 5e-6 |
| Stage 4: Refiner 训练 | Base model + Audio DiT | Video upsampler | 1e-4 |

```bash
# Stage 1: TA-CrossAttn 暖启动
bash scripts/train_stage1_warmup.sh

# Stage 2: 联合微调
bash scripts/train_stage2_joint.sh

# Stage 3: 全模型微调
bash scripts/train_stage3_finetune.sh
```

## 项目结构

```
alive-wan2x/
├── configs/              # 配置文件 (OmegaConf YAML)
│   ├── model/           # 模型配置
│   └── train/           # 训练配置
├── models/
│   ├── video_dit/       # Wan2.x Video DiT 适配
│   ├── audio_dit/       # Audio DiT 实现
│   ├── joint/           # 联合 AV DiT, TA-CrossAttn, UniTemp-RoPE
│   └── text_encoders/   # 文本编码器 (Qwen3, Qwen2.5, T5)
├── data/                # 数据管线
├── training/            # 训练代码 (trainer, losses, schedulers)
├── inference/           # 推理代码 (pipeline, multi_cfg)
└── benchmarks/          # 评估基准
```

## 硬件需求

| 阶段 | 最低配置 | 推荐配置 |
|------|---------|---------|
| Audio DiT 预训练 | 8×A100 80G | 16×A100/H100 |
| Joint AV 训练 | 16×A100 80G | 32×H100 |
| 推理 (单卡) | 1×RTX 4090 24G | 1×A100 80G |

## 性能指标

| 指标 | 480P | 1080P |
|------|------|-------|
| FVD (视频) | TBD | TBD |
| FAD (音频) | TBD | TBD |
| Sync Score | TBD | TBD |
| 推理速度 | ~5min/clip | ~15min/clip |

## 许可证

Apache 2.0

## 引用

```bibtex
@misc{alive-wan2x,
  title={Alive on Wan2.x: Audio-Video Generation with State-of-the-Art Video Foundation Models},
  author={Your Name},
  year={2026}
}
```

## 相关链接

- [Alive 原始项目](https://github.com/FoundationVision/Alive)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)

---

</details>

<details>
<summary>🇬🇧 English</summary>

---

Alive audio-visual video generation model rebuilt on Wan2.1/Wan2.2. Generates videos with synchronized sound from text or image inputs.

## Features

- **T2VA** (Text-to-Video&Audio): Generate video with sound from text prompts
- **I2VA** (Image-to-Video&Audio): Generate video with sound from an image
- **R2VA** (Reference-to-Video&Audio): Animate a reference portrait with synchronized audio
- 480P base generation + 1080P cascaded refinement
- Arbitrary duration generation (5–10 seconds)
- Multi-condition CFG for fine-grained audio-visual sync control

## Architecture Overview

Built on the **Dual Stream + Single Stream** paradigm:

```
Input Layer
├── Video Latents (Wan-VAE)
├── Audio Latents (WavVAE)
└── Text Embeddings (Qwen3-8B/4B + Qwen2.5-32B optional)

Dual Stream Phase (M=16 blocks)
├── Video DiT Blocks  [B, T_v, 4096]
├── Audio DiT Blocks  [B, T_a, 1536]
└── TA-CrossAttn (time-aligned audio-visual cross-attention)

Single Stream Phase (N=40 blocks)
└── Joint audio-visual processing [B, T_v+T_a, 4096]
```

### Core Technologies

- **UniTemp-RoPE**: Unified temporal positional encoding for physical-time alignment between video (1/6s/token) and audio (1/50s/token) streams
- **TA-CrossAttn**: Time-Aligned Cross-Attention ensuring audio-visual synchronization
- **Dual Conditioning**: Speech Text (content) + Descriptive Prompt (style) dual-path control
- **Multi-CFG**: `v_guided = v_uncond + cfg_text*(v_text - v_uncond) + cfg_mutual*(v_cond - v_text)`
- **Cascaded Refinement**: 480P → 1080P high-resolution upsampling

## Quick Start

### Requirements

```
Python >= 3.10
CUDA >= 11.7
PyTorch >= 2.1
```

### Installation

```bash
# Clone the repo
git clone https://github.com/your-username/alive-wan2x.git
cd alive-wan2x

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Download model weights
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./models/wan2.1-t2v-14b
huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/qwen3-8b
# Or use the lighter 4B variant:
# huggingface-cli download Qwen/Qwen3-4B --local-dir ./models/qwen3-4b
```

### Inference

```bash
# Generate video with sound from text
python -m inference.pipeline \
    --prompt "A cat running through a garden" \
    --duration 5 \
    --resolution 480p \
    --output output.mp4

# Or use the CLI entry point
alive-infer --prompt "prompt text" --output output.mp4
```

## Training

### Data Preparation

Datasets should be in JSON/JSONL format with fields: `video_path`, `audio_path`, `caption`. Recommended datasets: VGGSound, AudioSet, LibriTTS, AudioCaps.

### Four-Stage Training Strategy

| Stage | Frozen | Trainable | LR |
|-------|--------|-----------|-----|
| 1: TA-CrossAttn warmup | Video DiT + Audio DiT | CrossAttn only | 1e-4 |
| 2: Joint fine-tuning | Video DiT + Text | Audio DiT + CrossAttn | 1e-5 |
| 3: Full model tuning | Text encoder | All params | 5e-6 |
| 4: Refiner training | Base model + Audio DiT | Video upsampler | 1e-4 |

```bash
# Stage 1: Warm up TA-CrossAttn
bash scripts/train_stage1_warmup.sh

# Stage 2: Joint fine-tuning
bash scripts/train_stage2_joint.sh

# Stage 3: Full model fine-tuning
bash scripts/train_stage3_finetune.sh
```

## Project Structure

```
alive-wan2x/
├── configs/              # Configuration files (OmegaConf YAML)
│   ├── model/           # Model configs
│   └── train/           # Training stage configs
├── models/
│   ├── video_dit/       # Wan2.x Video DiT adapter
│   ├── audio_dit/       # Audio DiT implementation
│   ├── joint/           # Joint AV DiT, TA-CrossAttn, UniTemp-RoPE
│   └── text_encoders/   # Text encoders (Qwen3, Qwen2.5, T5)
├── data/                # Data pipelines
├── training/            # Training code (trainer, losses, schedulers)
├── inference/           # Inference code (pipeline, multi_cfg)
└── benchmarks/          # Evaluation benchmarks
```

## Hardware Requirements

| Stage | Minimum | Recommended |
|-------|---------|-------------|
| Audio DiT pretraining | 8×A100 80G | 16×A100/H100 |
| Joint AV training | 16×A100 80G | 32×H100 |
| Inference (single GPU) | 1×RTX 4090 24G | 1×A100 80G |

## Performance

| Metric | 480P | 1080P |
|--------|------|-------|
| FVD (video) | TBD | TBD |
| FAD (audio) | TBD | TBD |
| Sync Score | TBD | TBD |
| Inference speed | ~5 min/clip | ~15 min/clip |

## License

Apache 2.0

## Citation

```bibtex
@misc{alive-wan2x,
  title={Alive on Wan2.x: Audio-Video Generation with State-of-the-Art Video Foundation Models},
  author={Your Name},
  year={2026}
}
```

## Related Links

- [Alive Original Project](https://github.com/FoundationVision/Alive)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)

---

</details>

</div>
