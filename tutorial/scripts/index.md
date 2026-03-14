# DreamZero Scripts 模块概览

## 文件概述

`scripts/` 目录包含 DreamZero 项目的所有可执行脚本，涵盖模型训练、数据转换和评估分析三大类功能。这些脚本是用户与 DreamZero 系统交互的主要入口，封装了复杂的配置细节，使得用户可以通过简单的命令行调用完成端到端的工作流。

## 目录结构

```
scripts/
├── train/                              # 训练脚本
│   ├── droid_training_lora.sh          # DROID 数据集 LoRA 微调
│   ├── droid_training_full_finetune.sh # DROID 数据集全参数微调
│   ├── agibot_training.sh             # AGIBot 数据集 LoRA 训练
│   └── yam_training.sh                # YAM 数据集 LoRA 训练
├── data/                               # 数据转换脚本
│   ├── convert_droid.py               # DROID RLDS → LeRobot 格式转换
│   ├── convert_lerobot_to_gear.py     # 通用 LeRobot → GEAR/DreamZero 格式转换
│   └── convert_agibot.py             # AGIBot 原始数据 → LeRobot 格式转换
├── compare_loss.py                     # LoRA vs 全参数微调 Loss 对比
└── open_loop_yam.py                   # YAM 开环评估
```

## 模块分类

### 训练脚本 (`scripts/train/`)

训练脚本统一使用 `torchrun` 进行分布式训练，调用 `groot/vla/experiment/experiment.py` 作为训练入口。所有脚本共享以下特征：

- 基于 Hydra 配置系统传递训练参数
- 自动下载模型权重（Wan2.1-I2V-14B-480P、umt5-xxl）
- 支持 DeepSpeed ZeRO-2 分布式优化
- 使用 BF16/TF32 混合精度训练

详见 [训练脚本概览](train/index.md)。

### 数据转换脚本 (`scripts/data/`)

数据转换脚本负责将不同来源的机器人数据集统一转换为 DreamZero 训练所需的 LeRobot v2 格式，包括 parquet 数据文件、MP4 视频文件以及元数据（modality.json、stats.json 等）。

详见 [数据转换脚本概览](data/index.md)。

### 工具脚本

| 脚本 | 用途 |
|------|------|
| `compare_loss.py` | 对比 LoRA 和全参数微调的训练损失曲线，输出表格和可视化图表 |
| `open_loop_yam.py` | 在 YAM 数据集上进行离线开环评估，计算预测动作与真实动作的 MSE |

## 核心依赖关系

```
scripts/train/*.sh
    └── groot/vla/experiment/experiment.py  (训练入口)
        ├── groot/vla/configs/              (Hydra 配置)
        ├── groot/vla/model/                (模型定义)
        └── groot/vla/data/                 (数据加载)

scripts/data/*.py
    └── 生成 LeRobot v2 格式数据
        └── 被 groot/vla/data/dataset/ 加载

scripts/open_loop_yam.py
    └── groot/vla/model/n1_5/sim_policy.py  (推理策略)
```

## 典型工作流

1. **数据准备**：使用 `scripts/data/convert_*.py` 将原始数据转换为 LeRobot 格式
2. **模型训练**：使用 `scripts/train/*.sh` 启动分布式训练
3. **损失对比**：使用 `scripts/compare_loss.py` 对比不同训练策略的效果
4. **开环评估**：使用 `scripts/open_loop_yam.py` 评估模型的动作预测精度

## 总结

`scripts/` 模块是 DreamZero 的用户操作层，将底层的模型、数据和训练逻辑封装为简洁易用的命令行工具。训练脚本通过 Shell 封装了 `torchrun` + Hydra 的完整配置；数据脚本处理多种机器人数据源的格式统一；工具脚本提供训练分析和模型评估能力。理解这些脚本的参数和工作原理，是高效使用 DreamZero 系统的关键。
