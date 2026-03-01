# Alive on Wan2.x 技术文档

本文档详细介绍 Alive on Wan2.x 的技术架构和实现细节。

## 目录

1. [总体架构](#总体架构)
2. [核心模块](#核心模块)
3. [训练策略](#训练策略)
4. [推理流程](#推理流程)

## 总体架构

### Dual Stream + Single Stream 范式

Alive 采用两阶段架构设计：

```
                    ┌─────────────────┐
                    │   输入层        │
                    │ Video+Audio+Text│
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
    ┌───────▼───────┐ ┌──────▼──────┐ ┌───────▼───────┐
    │ Video DiT     │ │ Audio DiT   │ │ Text Encoder  │
    │ (Wan2.x based)│ │ (32 blocks) │ │ T5+Qwen       │
    └───────┬───────┘ └──────┬──────┘ └───────┬───────┘
            │                │                │
            └────────────────┼────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Dual Stream    │
                    │  (M=16 blocks)  │
                    │ TA-CrossAttn    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Single Stream  │
                    │  (N=40 blocks)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   输出层        │
                    │ Video+Audio     │
                    └─────────────────┘
```

## 核心模块

### UniTemp-RoPE

统一时间位置编码，将视频帧和音频 token 映射到共享的物理时间坐标系。

**关键参数：**
- 视频：24fps，VAE 时间压缩比 4:1 → 每个 token = 4/24 = 1/6 秒
- 音频：取决于 Audio VAE 的压缩比

### TA-CrossAttn

时间对齐交叉注意力模块，实现音视频双向交互：
- Audio→Video：音频关注视频获取视觉语境
- Video→Audio：视频关注音频获取声音信号

### 双条件控制

Audio DiT 的双路条件控制：
1. **Speech Text**：控制说话内容（说什么）
2. **Descriptive Prompt**：控制声音风格（怎么说/什么声音）

## 训练策略

### 四阶段训练流程

| 阶段 | 目的 | 学习率 | 训练参数 |
|------|------|--------|---------|
| Stage 1 | TA-CrossAttn 暖启动 | 1e-4 | 仅 CrossAttn |
| Stage 2 | 联合微调 | 1e-5 (Audio) | Audio DiT+CrossAttn |
| Stage 3 | 全模型微调 | 5e-6 | 所有参数 |
| Stage 4 | Refiner 训练 | - | Video Upsampler |

### 关键技巧

1. **非对称学习率**：Audio DiT 学习率低于其他模块 5-10 倍
2. **EMA 权重**：Audio DiT 使用更高 decay (0.9999)
3. **渐进式数据混合**：逐步增加音视频数据比例

## 推理流程

### 多条件 CFG

分别控制文本条件和跨模态注意力：

```python
v_guided = v_uncond \
    + cfg_scale_text * (v_text_only - v_uncond) \
    + cfg_scale_mutual * (v_cond - v_text_only)
```

**推荐参数：**
- `cfg_scale_text`: 7.5
- `cfg_scale_mutual`: 2.0

## 性能优化

- TeaCache: ~2x 加速
- FP8 量化：降低显存
- FlashAttention 3: Hopper 架构加速
- Ulysses 并行：多 GPU 序列并行
