# 元数据工具 (`groot/vla/data/dataset/metadata.py`)

## 文件概述

`metadata.py` 原本提供了数据集元数据的生成和管理功能，包括统计信息计算和元数据文件的读写。但目前该文件中的所有代码已被**完全注释掉**，相关功能已迁移到 `lerobot.py` 中。

## 关键代码解析

### 注释代码中的核心功能（已弃用）

#### 1. `calculate_dataset_statistics` (已弃用)

```python
# def calculate_dataset_statistics(parquet_paths, features=None):
#     """计算 parquet 文件的统计信息"""
#     # 收集所有数据 -> 计算均值、标准差、最小值、最大值、分位数
```

此函数已被 `lerobot.py` 中同名函数取代。

#### 2. `get_metadata` (已弃用)

```python
# def get_metadata(embodiment_tag, metadata_version, ...):
#     """获取指定 embodiment 标签和版本的元数据"""
```

根据 `EmbodimentTag` 加载或生成数据集元数据的函数。文件中存在两个版本的 `get_metadata` 函数定义（均已注释），反映了 API 的演进历史。

#### 3. `generate_metadata` (已弃用)

```python
# def generate_metadata(embodiment_tag, dataset_paths, le_modality_meta, ...):
#     """生成完整的数据集元数据"""
```

该函数负责将 LeRobot 原始元数据转换为内部使用的 `DatasetMetadata` 格式。其核心逻辑包括：

- 解析 state/action 模态的连续性和旋转类型
- 提取视频模态的分辨率、通道数和帧率
- 处理标注（annotation）元数据
- 调用 `calculate_dataset_statistics` 计算统计信息

## 核心类/函数表格

| 名称 | 类型 | 状态 | 说明 |
|------|------|------|------|
| `calculate_dataset_statistics` | 函数 | 已弃用 | 统计信息计算，已迁移至 `lerobot.py` |
| `get_metadata` | 函数 | 已弃用 | 元数据加载/生成，已迁移至 `lerobot.py` |
| `generate_metadata` | 函数 | 已弃用 | 元数据生成，已迁移至 `lerobot.py` |

## 与其他模块的关系

- 原本依赖 `schema` 中的 `EmbodimentTag`、`LeRobotModalityMetadata` 等
- 原本依赖 `macro.py` 中的常量
- 原本依赖 `registry.py` 中的数据集注册表
- 功能已全部迁移到 `lerobot.py` 的 `LeRobotSingleDataset._get_metadata()` 方法

## 总结

`metadata.py` 是一个已弃用的历史文件，保留了项目早期元数据管理方案的代码（均已注释）。当前版本的元数据处理逻辑已集成到 `lerobot.py` 中的 `LeRobotSingleDataset` 类。阅读此文件有助于理解 DreamZero 数据管线的演进历史。
