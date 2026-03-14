# 数据配置目录索引

## 概述

`groot/vla/configs/data/` 包含 DreamZero VLA 训练所需的数据集配置文件。所有配置基于 Hydra 框架，通过 YAML 文件定义数据路径、模态设置、采样参数等。

## 子目录

| 目录 | 说明 |
|------|------|
| [dreamzero/](dreamzero/index.md) | DreamZero 数据集配置（DROID、AGIBot、YAM 等） |

## 配置文件列表

### dreamzero/

- [base_48_wan_fine_aug_relative.yaml](dreamzero/base_48_wan_fine_aug_relative.md) -- 基础数据管道配置（48 帧、Wan 精细增强、相对动作）
- [droid_relative.yaml](dreamzero/droid_relative.md) -- DROID 数据集相对动作配置
- [agibot_relative.yaml](dreamzero/agibot_relative.md) -- AGIBot 数据集相对动作配置
- [yam_relative.yaml](dreamzero/yam_relative.md) -- YAM 数据集相对动作配置
