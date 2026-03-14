# 数据集子模块 (`groot/vla/data/dataset`)

## 文件概述

`dataset` 子模块负责数据集的定义、加载与管理，是 DreamZero 训练管线的核心数据入口。它围绕 LeRobot 数据格式构建，提供了单数据集、混合数据集、分片数据集等多种加载方式。

## 文件列表

| 文件 | 说明 | 教程链接 |
|------|------|----------|
| `lerobot.py` | LeRobot 数据集核心实现，包含单数据集和混合数据集 | [lerobot.md](lerobot.md) |
| `lerobot_sharded.py` | 分片数据集实现，支持大规模数据集的分布式训练 | [lerobot_sharded.md](lerobot_sharded.md) |
| `macro.py` | 常量定义，包括各类元数据文件名 | [macro.md](macro.md) |
| `metadata.py` | 元数据生成工具（已弃用，代码已注释） | [metadata.md](metadata.md) |
| `registry.py` | 数据集注册表，映射 embodiment 标签到数据路径 | [registry.md](registry.md) |

## `__init__.py`

```python
from .lerobot import ModalityConfig
```

导出 `ModalityConfig` 类，用于配置各模态（视频、状态、动作等）的采样参数。

## 模块依赖关系

- `lerobot.py` 依赖 `macro.py` 中的常量定义和 `schema` 模块中的数据模式
- `lerobot_sharded.py` 继承 `lerobot.py` 中的基类
- `registry.py` 依赖 `schema` 中的 `EmbodimentTag`
- `metadata.py` 中的功能已迁移到 `lerobot.py` 中
