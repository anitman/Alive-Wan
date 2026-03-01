# Wan2.x 有声视频生成模型

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

Dual Stream 阶段
├── Video DiT Blocks
├── Audio DiT Blocks  
└── TA-CrossAttn (音视频跨模态注意力)

Single Stream 阶段
└── 联合音视频处理
```

### 核心技术

- **UniTemp-RoPE**：统一时间位置编码，实现音视频物理时间对齐
- **TA-CrossAttn**：时间对齐交叉注意力，确保音视频同步
- **双条件控制**：Speech Text + Descriptive Prompt 双路控制
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
```

## 训练

### 数据准备

```bash
# 下载数据集
# VGGSound, AudioSet, LibriTTS, AudioCaps 等
```

### 多阶段训练

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
├── configs/              # 配置文件
│   ├── model/           # 模型配置
│   └── train/           # 训练配置
├── models/
│   ├── video_dit/       # Wan2.x Video DiT 适配
│   ├── audio_dit/       # Audio DiT 实现
│   ├── joint/           # 联合 AV DiT, TA-CrossAttn, UniTemp-RoPE
│   └── text_encoders/   # 文本编码器
├── data/                # 数据管线
├── training/            # 训练代码
├── inference/           # 推理代码
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
| FID (音频) | TBD | TBD |
| Sync Score | TBD | TBD |
| 推理速度 | ~5min/clip | ~15min/clip |

## 技术文档

详细技术架构和实现方案请参考 [技术文档](docs/TECHNICAL.md)

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