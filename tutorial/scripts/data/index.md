# DreamZero 数据转换脚本概览

## 文件概述

`scripts/data/` 目录包含将不同来源的机器人操作数据集转换为 DreamZero 训练所需格式的脚本。DreamZero 的训练流水线基于 LeRobot v2 数据格式，并在此基础上扩展了 GEAR 特有的元数据文件（如 `modality.json`、`embodiment.json`、`relative_stats_dreamzero.json`）。

## 脚本列表

| 脚本 | 源格式 | 目标格式 | 主要功能 |
|------|--------|----------|---------|
| `convert_droid.py` | DROID RLDS/TFDS | LeRobot v2 | 将 DROID 1.0.1 数据集转换为 LeRobot 格式，支持空闲帧过滤 |
| `convert_lerobot_to_gear.py` | LeRobot v2 | GEAR/DreamZero | 为已有 LeRobot 数据集生成 DreamZero 训练所需的元数据 |
| `convert_agibot.py` | AGIBot HDF5 | LeRobot v2 | 将 AGIBot 原始数据（新旧两种格式）转换为 LeRobot 格式 |

## 数据格式概述

### LeRobot v2 格式结构

DreamZero 使用的标准数据格式如下：

```
dataset_root/
├── meta/
│   ├── info.json              # 数据集基本信息（episode 数、帧数、FPS 等）
│   ├── modality.json          # 状态/动作/视频/标注的字段映射
│   ├── embodiment.json        # 机器人形态标签
│   ├── tasks.jsonl            # 任务描述列表
│   ├── episodes.jsonl         # 各 episode 的元数据
│   ├── stats.json             # 数据集统计信息（均值、标准差等）
│   └── relative_stats_dreamzero.json  # 相对动作统计
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
│   └── chunk-001/
│       └── ...
└── videos/
    ├── chunk-000/
    │   ├── observation.images.camera_1/
    │   │   ├── episode_000000.mp4
    │   │   └── ...
    │   └── observation.images.camera_2/
    │       └── ...
    └── chunk-001/
        └── ...
```

### 关键元数据文件

| 文件 | 用途 |
|------|------|
| `modality.json` | 定义状态/动作向量中各子字段的起止索引、数据类型和原始列名 |
| `embodiment.json` | 标记机器人形态类型，训练流水线据此选择对应的数据配置 |
| `stats.json` | 存储各数据列的统计量（mean/std/min/max/q01/q99），用于归一化 |
| `relative_stats_dreamzero.json` | 存储相对动作的统计量，用于相对动作空间的归一化 |

## 转换流程

### 从原始数据到训练数据的典型路径

```
DROID RLDS 数据
    │
    ▼ (convert_droid.py)
LeRobot v2 格式（含 modality.json）
    │
    ▼ (训练脚本直接使用)
DreamZero 训练

AGIBot HDF5 数据
    │
    ▼ (convert_agibot.py)
LeRobot v2 格式
    │
    ▼ (convert_lerobot_to_gear.py，可选)
GEAR/DreamZero 扩展元数据
    │
    ▼ (训练脚本使用)
DreamZero 训练

自定义 LeRobot v2 数据
    │
    ▼ (convert_lerobot_to_gear.py)
GEAR/DreamZero 扩展元数据
    │
    ▼ (训练脚本使用)
DreamZero 训练
```

## 核心函数/类表格

| 脚本 | 核心函数/类 | 功能 |
|------|-----------|------|
| `convert_droid.py` | `convert_droid_dataset()` | 主转换流程：过滤、分片处理、写入 |
| `convert_droid.py` | `process_tfrecord()` / `process_sample()` | 单个分片/样本的处理逻辑 |
| `convert_droid.py` | `encode_video()` | 使用 PyAV 编码视频 |
| `convert_lerobot_to_gear.py` | `build_modality_json()` | 构建 modality.json 结构 |
| `convert_lerobot_to_gear.py` | `compute_stats()` / `compute_relative_stats()` | 计算数据集统计量 |
| `convert_agibot.py` | `AgiBotDataset` | 继承 LeRobotDataset，定制保存逻辑 |
| `convert_agibot.py` | `load_local_dataset_old_format()` / `load_local_dataset_new_format()` | 加载新旧格式 AGIBot 数据 |

## 与其他模块的关系

- **数据加载**：转换后的数据被 `groot/vla/data/dataset/lerobot.py` 加载
- **数据配置**：`groot/vla/configs/data/dreamzero/` 下的 YAML 配置文件引用转换后的数据路径
- **训练脚本**：`scripts/train/*.sh` 通过环境变量指定转换后的数据目录

## 总结

数据转换脚本是 DreamZero 数据流水线的关键环节，负责将异构的机器人数据源统一为标准的 LeRobot v2 + GEAR 扩展格式。`convert_droid.py` 和 `convert_agibot.py` 处理特定数据源的转换，`convert_lerobot_to_gear.py` 则提供通用的元数据生成能力，适用于任何已有的 LeRobot v2 数据集。
