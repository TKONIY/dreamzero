# `dreamzero/` -- DreamZero 核心模型实现

## 模块概述

DreamZero 核心模块实现了基于 Wan2.1 视频生成模型的 VLA (Vision-Language-Action) 架构，将世界模型（视频生成）和策略模型（动作预测）统一在同一个扩散 Transformer 中。

## 核心文件

| 文件 | 描述 |
|------|------|
| [`base_vla.py`](base_vla.md) | VLA 模型基类，组合 backbone + action_head，支持多种推理模式和权重加载方式 |

## 子模块

| 目录 | 描述 |
|------|------|
| [`action_head/`](action_head/index.md) | Flow Matching 动作头：基于 Wan2.1 DiT 的联合视频-动作生成 |
| [`backbone/`](backbone/index.md) | Backbone 模块：提供 IdentityBackbone（透传 backbone） |
| [`transform/`](transform/index.md) | 数据变换：多形态数据统一化处理 |
| [`modules/`](modules/index.md) | 核心神经网络组件：DiT、VAE、编码器、调度器、显存管理等 |

## 核心创新

1. **视频-动作联合生成**: 将动作 token 嵌入视频扩散过程
2. **因果式自回归生成**: 块级因果注意力支持逐帧生成
3. **多机器人形态**: CategorySpecificMLP 支持跨形态训练
4. **Teacher Forcing**: clean/noisy 分割稳定训练

## `__init__.py`

`__init__.py` 文件为空。
