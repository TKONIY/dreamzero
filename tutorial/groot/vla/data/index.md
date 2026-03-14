# DreamZero 数据模块 (`groot/vla/data`)

## 文件概述

`groot/vla/data` 是 DreamZero 项目中负责数据加载、预处理和转换的核心模块。它为视觉-语言-动作 (VLA) 模型的训练与推理提供了完整的数据管线支持，主要包含以下四个子模块：

## 子模块概览

| 子模块 | 路径 | 功能说明 |
|--------|------|----------|
| [dataset](dataset/index.md) | `groot/vla/data/dataset/` | 数据集定义与加载，包括 LeRobot 格式数据集、分片数据集、混合数据集等 |
| [schema](schema/index.md) | `groot/vla/data/schema/` | 数据模式定义，包括 embodiment 标签枚举和 LeRobot 元数据 schema |
| [transform](transform/index.md) | `groot/vla/data/transform/` | 数据变换管线，包含视频增强、状态/动作归一化、语言 tokenization、拼接等 |
| [conversion](conversion/index.md) | `groot/vla/data/conversion/` | 数据格式转换工具，将原始 GR1 数据转换为可训练格式 |

## 模块关系

```
schema (数据模式定义)
  |
  v
dataset (数据集加载) ---> transform (数据变换管线)
  |
  v
conversion (格式转换工具)
```

- **schema** 定义了所有数据结构（`EmbodimentTag`、`DatasetMetadata` 等），是其他模块的基础依赖。
- **dataset** 负责从磁盘加载 LeRobot 格式的数据，生成 PyTorch `Dataset` 实例，并管理元数据和统计信息。
- **transform** 提供可组合的数据变换管线，对数据集输出的原始数据进行增强和标准化处理。
- **conversion** 包含将 GR1 机器人原始数据转换为 LeRobot 格式的辅助工具。

## `__init__.py`

`groot/vla/data/__init__.py` 为空文件，仅作为包标识。
