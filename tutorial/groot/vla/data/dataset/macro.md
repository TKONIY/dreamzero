# 常量定义 (`groot/vla/data/dataset/macro.py`)

## 文件概述

`macro.py` 定义了 DreamZero 数据管线中使用的各种文件名常量。这些常量被其他模块广泛引用，用于定位数据集目录中的元数据文件和数据文件。

## 关键代码解析

### 1. 已弃用的常量

```python
METADATA_FILENAME = "metadata.json"
DATA_FILENAME = "dataset.hdf5"
FULL_SET_NAME = "FullSet"
```

这些常量来自旧版数据格式，目前已弃用，保留仅用于向后兼容。

### 2. LeRobot 格式常量

```python
LE_ROBOT_METADATA_FILENAME = "metadata.json"       # 元数据文件
LE_ROBOT_MODALITY_FILENAME = "modality.json"        # 模态配置文件
LE_ROBOT_FEATURES_FILENAME = "features.json"        # 特征定义文件
LE_ROBOT_STATISTICS_FILENAME = "stats.json"          # 统计信息文件
LE_ROBOT_EMBODIMENT_FILENAME = "embodiment.json"    # Embodiment 配置文件
```

### 3. LeRobot 扩展常量

```python
LE_ROBOT_EPISODE_FILENAME = "episodes.jsonl"        # 剧集信息（JSONL 格式）
LE_ROBOT_TASKS_FILENAME = "tasks.jsonl"              # 任务描述
LE_ROBOT_INFO_FILENAME = "info.json"                 # 数据集基本信息
LE_ROBOT_METADATA_DIR = "meta"                       # 元数据目录名
```

这些是 LeRobot 格式的扩展定义，`meta` 目录下存放所有元数据文件。

## 核心类/函数表格

| 名称 | 类型 | 说明 |
|------|------|------|
| `LE_ROBOT_MODALITY_FILENAME` | 常量 | 模态配置文件名 `"modality.json"` |
| `LE_ROBOT_EPISODE_FILENAME` | 常量 | 剧集信息文件名 `"episodes.jsonl"` |
| `LE_ROBOT_INFO_FILENAME` | 常量 | 数据集信息文件名 `"info.json"` |
| `LE_ROBOT_METADATA_DIR` | 常量 | 元数据目录名 `"meta"` |

## 与其他模块的关系

- **被 `lerobot.py` 引用** - 用于定位元数据文件路径
- **被 `conversion/gr1/get_initial_actions.py` 引用** - 用于定位 LeRobot 数据集中的元数据

## 总结

`macro.py` 是一个简单的常量定义文件，集中管理了 LeRobot 数据格式中的所有文件名约定。它的存在使得文件路径的修改只需在一处完成，提高了代码的可维护性。注意该文件与 `lerobot.py` 中也定义了一套类似的常量（带完整路径前缀 `meta/`），两者有一定重叠。
