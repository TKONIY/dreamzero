# DreamZero VLA 配置模块索引

## 概述

`groot/vla/configs/` 目录包含 DreamZero VLA（Vision-Language-Action）系统的全部配置文件。基于 [Hydra](https://hydra.cc/) 配置框架，采用分层组合式设计，将模型定义、数据管道、训练超参数和分布式策略等配置解耦为独立模块。

## 目录结构

```
configs/
├── conf.yaml                          # 顶层主配置文件
├── model/dreamzero/                   # 模型配置
│   ├── vla.yaml                       # VLA 模型入口
│   ├── action_head/                   # 动作头配置
│   ├── backbone/                      # 骨干网络配置
│   └── transform/                     # 数据变换配置
├── data/dreamzero/                    # 数据集配置
│   ├── base_48_wan_fine_aug_relative.yaml  # 基础数据管道
│   ├── droid_relative.yaml            # DROID 数据集
│   ├── agibot_relative.yaml           # AGIBot 数据集
│   └── yam_relative.yaml             # YAM 数据集
└── deepspeed/                         # DeepSpeed 分布式训练配置
    ├── zero2.json                     # ZeRO Stage 2
    ├── zero2_offload.json             # ZeRO Stage 2 + CPU 卸载
    └── zero3.json                     # ZeRO Stage 3
```

## 配置文件索引

### 主配置
- [conf.yaml](conf.md) -- 顶层训练配置，组合所有子模块

### 模型配置
- [model/dreamzero/](model/dreamzero/index.md) -- DreamZero 模型配置目录

### 数据配置
- [data/dreamzero/](data/dreamzero/index.md) -- 数据集与数据管道配置目录

### 分布式训练配置
- [deepspeed/](deepspeed/index.md) -- DeepSpeed ZeRO 优化配置目录

## 配置组合关系

```
conf.yaml
  ├─ model: dreamzero/vla.yaml
  │    ├─ backbone: identity.yaml
  │    ├─ action_head: wan_flow_matching_action_tf.yaml
  │    └─ transform: dreamzero_cotrain.yaml
  │         └─ base.yaml (体型映射)
  ├─ data: dreamzero/*_relative.yaml
  │    └─ base_48_wan_fine_aug_relative.yaml
  └─ deepspeed: zero2.json / zero2_offload.json / zero3.json
```

## 快速开始

启动 DreamZero 训练的最小命令示例：

```bash
python train.py \
  wandb_project=my_project \
  output_dir=./output \
  data=dreamzero/droid_relative \
  droid_data_root=/path/to/droid \
  training_args.deepspeed=groot/vla/configs/deepspeed/zero2.json
```
