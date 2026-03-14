# DreamZero 训练脚本概览

## 文件概述

`scripts/train/` 目录下包含 DreamZero 项目在不同数据集和训练策略下的训练启动脚本。所有脚本均为 Bash Shell 脚本，封装了 `torchrun` 分布式训练的完整配置，用户只需设置数据路径和输出目录即可启动训练。

## 脚本列表

| 脚本 | 数据集 | 训练策略 | GPU 需求 | 默认训练步数 |
|------|--------|----------|----------|-------------|
| `droid_training_lora.sh` | DROID | LoRA 微调 | 8 GPU | 100 |
| `droid_training_full_finetune.sh` | DROID | 全参数微调 | 8x H100 | 100 |
| `agibot_training.sh` | AGIBot | LoRA 微调 | 自动检测 | 5,000 |
| `yam_training.sh` | YAM | LoRA 微调 | 自动检测 | 100,000 |

## 统一架构

所有训练脚本遵循相同的结构模式：

```
1. 环境变量设置（HYDRA_FULL_ERROR=1）
2. 用户配置区（数据路径、输出目录、GPU 数量、模型权重路径）
3. 自动下载模型权重（Wan2.1-I2V-14B-480P、umt5-xxl）
4. 数据集路径验证
5. torchrun 启动训练（传递 Hydra 配置参数）
```

## 共享配置参数

以下核心参数在所有脚本中一致：

| 参数 | 值 | 说明 |
|------|-----|------|
| `model` | `dreamzero/vla` | 使用 DreamZero VLA 模型 |
| `model/dreamzero/action_head` | `wan_flow_matching_action_tf` | 基于 Wan 的 Flow Matching 动作头 |
| `model/dreamzero/transform` | `dreamzero_cotrain` | DreamZero 联合训练变换 |
| `num_frames` | 33 | 输入视频帧数 |
| `action_horizon` | 24 | 动作预测时间窗口 |
| `num_views` | 3 | 相机视角数量 |
| `num_frame_per_block` | 2 | 每个 block 的帧数 |
| `num_action_per_block` | 24 | 每个 block 的动作数 |
| `num_state_per_block` | 1 | 每个 block 的状态数 |
| `bf16` / `tf32` | true | 混合精度训练 |
| `image_resolution_width` | 320 | 图像宽度 |
| `image_resolution_height` | 176 | 图像高度 |
| `frame_seqlen` | 880 | 帧序列长度 |
| `max_chunk_size` | 4 | 最大分块大小 |

## 关键差异对比

### LoRA vs 全参数微调

| 配置项 | LoRA | 全参数微调 |
|--------|------|-----------|
| `train_architecture` | `lora` | `full` |
| `learning_rate` | `1e-4`（DROID）/ `1e-5`（其他） | `1e-5` |
| `deepspeed` | `zero2.json` | `zero2_offload.json`（含 CPU Offload） |
| `save_lora_only` | `true` | `false` |

### 跨数据集迁移训练

AGIBot 和 YAM 训练脚本相比 DROID 额外使用了以下参数：

- `pretrained_model_path`：加载预训练的 DreamZero-AgiBot 检查点
- `++action_head_cfg.config.skip_component_loading=true`：跳过动作头组件加载
- `++action_head_cfg.config.defer_lora_injection=true`：延迟 LoRA 注入

这使得模型可以先加载基础权重，再根据新数据集调整动作头配置。

## 模型权重依赖

所有脚本依赖以下预训练模型（自动下载）：

| 模型 | 来源 | 用途 |
|------|------|------|
| Wan2.1-I2V-14B-480P | `Wan-AI/Wan2.1-I2V-14B-480P` | DiT 视频生成骨干网络 |
| umt5-xxl | `google/umt5-xxl` | 文本编码器分词器 |
| DreamZero-AgiBot | `GEAR-Dreams/DreamZero-AgiBot` | 预训练检查点（AGIBot/YAM） |

## 与其他模块的关系

- **训练入口**：`groot/vla/experiment/experiment.py`
- **Hydra 配置**：`groot/vla/configs/` 下的 YAML 文件
- **DeepSpeed 配置**：`groot/vla/configs/deepspeed/zero2.json` 和 `zero2_offload.json`
- **数据配置**：`groot/vla/configs/data/dreamzero/` 下各数据集的配置
- **数据准备**：`scripts/data/` 下的转换脚本

## 总结

训练脚本是 DreamZero 训练流水线的最外层封装，通过统一的结构设计和灵活的参数配置，支持多数据集、多训练策略的快速切换。理解脚本中各参数的含义和差异，有助于用户根据自身需求定制训练方案。
