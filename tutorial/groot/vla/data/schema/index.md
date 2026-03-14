# 数据模式子模块 (`groot/vla/data/schema`)

## 文件概述

`schema` 子模块定义了 DreamZero 数据管线中使用的所有数据结构和模式。它基于 Pydantic 构建，提供了类型安全的数据验证和序列化能力。

## 文件列表

| 文件 | 说明 | 教程链接 |
|------|------|----------|
| `embodiment_tags.py` | Embodiment 标签枚举，标识不同机器人构型 | [embodiment_tags.md](embodiment_tags.md) |
| `lerobot.py` | LeRobot 数据模式定义，包含模态元数据和统计信息 | [lerobot.md](lerobot.md) |

## `__init__.py`

```python
from .embodiment_tags import EmbodimentTag
from .lerobot import (
    DatasetMetadata, DatasetModalities, DatasetStatisticalValues,
    DatasetStatistics, LeRobotActionMetadata, LeRobotModalityField,
    LeRobotModalityMetadata, LeRobotStateActionMetadata,
    LeRobotStateMetadata, RotationType, StateActionMetadata, VideoMetadata,
)
```

将所有核心数据结构统一导出，方便其他模块导入。
