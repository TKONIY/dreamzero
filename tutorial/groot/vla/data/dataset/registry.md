# 数据集注册表 (`groot/vla/data/dataset/registry.py`)

## 文件概述

`registry.py` 定义了数据集注册表，用于建立 `EmbodimentTag`（机器人构型标签）到本地数据集路径的映射关系。该文件提供了双向映射能力，是数据集发现和加载的配置入口。

## 关键代码解析

### 正向映射：Embodiment 标签到数据路径

```python
EMBODIMENT_TAGS_TO_DATASET_PATHS: dict[EmbodimentTag, list[Path]] = {}
```

这是一个空字典，需要用户根据自己的数据集存放位置进行填充。每个 `EmbodimentTag` 可以对应多个数据集路径（`list[Path]`），支持同一构型下的多数据集场景。

用法示例（需用户自行配置）：

```python
EMBODIMENT_TAGS_TO_DATASET_PATHS = {
    EmbodimentTag.REAL_GR1_ARMS_ONLY: [
        Path("/data/gr1_arms_dataset_v1"),
        Path("/data/gr1_arms_dataset_v2"),
    ],
}
```

### 反向映射：数据路径到 Embodiment 标签

```python
DATASET_PATHS_TO_EMBODIMENT_TAGS = {
    path: dataset_tag
    for dataset_tag, dataset_paths in EMBODIMENT_TAGS_TO_DATASET_PATHS.items()
    for path in dataset_paths
}
```

通过字典推导式自动生成反向映射，将每个数据集路径映射回对应的 `EmbodimentTag`。

## 核心类/函数表格

| 名称 | 类型 | 说明 |
|------|------|------|
| `EMBODIMENT_TAGS_TO_DATASET_PATHS` | 字典 | `EmbodimentTag -> list[Path]`，正向映射 |
| `DATASET_PATHS_TO_EMBODIMENT_TAGS` | 字典 | `Path -> EmbodimentTag`，反向映射 |

## 与其他模块的关系

- **依赖 `schema`** - 使用 `EmbodimentTag` 枚举
- **被 `metadata.py` 引用（已弃用）** - 原来的元数据生成流程使用此注册表查找数据集路径
- **用户配置入口** - 用户需要根据数据存放位置修改此文件

## 总结

`registry.py` 是一个轻量级的配置文件，提供了 embodiment 标签和数据集路径之间的双向映射。默认为空，需要用户根据实际环境配置。在当前版本中，`LeRobotSingleDataset` 通常直接接收 `dataset_path` 参数，因此此注册表主要用于批量数据管理场景。
