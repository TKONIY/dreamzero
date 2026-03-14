# LeRobot 数据集核心实现 (`groot/vla/data/dataset/lerobot.py`)

## 文件概述

`lerobot.py` 是 DreamZero 数据管线中最核心的文件，实现了基于 LeRobot 格式的数据集加载逻辑。它定义了三个主要类：`ModalityConfig`（模态配置）、`LeRobotSingleDataset`（单数据集）和 `LeRobotMixtureDataset`（混合数据集），并提供了统计信息计算、相对动作统计等辅助功能。

本文件约 1200+ 行代码，是整个数据模块中最庞大的文件。

## 关键代码解析

### 1. 常量定义

文件开头定义了大量路径常量，用于定位 LeRobot 数据集中的元数据文件：

```python
LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"    # 模态元数据
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"      # 剧集信息
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"            # 任务描述
LE_ROBOT_INFO_FILENAME = "meta/info.json"               # 数据集基本信息
LE_ROBOT_STATS_FILENAME = "meta/stats.json"             # 统计信息
LEROBOT_RELATIVE_STATS_FILE_NAME = "meta/relative_stats_dreamzero.json"  # 相对动作统计
```

### 2. `calculate_dataset_statistics` 函数

遍历所有 parquet 文件并计算指定特征的统计值（均值、标准差、最小值、最大值、1%/99% 分位数）：

```python
def calculate_dataset_statistics(parquet_paths, features=None):
    # 收集所有 parquet 数据 -> 合并 -> 按特征计算统计
    for le_modality in features:
        np_data = np.vstack([np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]])
        dataset_statistics[le_modality] = DatasetStatisticalValues(
            mean=..., std=..., min=..., max=..., q01=..., q99=...
        )
```

这个函数在统计文件不存在时会被自动调用，计算结果会缓存到磁盘。

### 3. `ModalityConfig` 类

使用 Pydantic 定义模态配置，控制每种模态（视频/状态/动作）的数据采样方式：

```python
class ModalityConfig(BaseModel):
    delta_indices: list[int]       # 相对于当前索引的偏移量列表
    eval_delta_indices: list[int] | None = None  # 评估时使用的偏移量
    modality_keys: list[str]       # 该模态下需要加载的键列表
```

`delta_indices` 是核心设计：例如 `delta_indices=[0, 1, 2, 3]` 表示从当前时间步开始加载连续 4 帧数据，这对于动作预测的 horizon 设计至关重要。

### 4. `LeRobotSingleDataset` 类

继承自 PyTorch `Dataset`，实现了完整的单数据集加载逻辑。

#### 初始化流程

`__init__` 方法按以下顺序完成初始化：

1. **路径与配置设置** - 验证数据路径、设置 embodiment 标签
2. **元数据加载** - 加载 modality、info、stats 等元数据文件
3. **轨迹信息提取** - 从 `episodes.jsonl` 获取轨迹 ID 和长度
4. **相对动作统计** - 如果启用 `relative_action`，加载或计算相对动作的统计值
5. **数据集元数据构建** - 将 LeRobot 格式的元数据转换为内部 `DatasetMetadata` 格式
6. **变换管线设置** - 将元数据传递给变换管线
7. **完整性检查** - 验证数据集的合法性

#### 相对动作统计

DreamZero 的一个重要特性是支持相对动作（relative action），即动作值相对于当前状态的偏移量。相关方法包括：

- `_get_lerobot_relative_stats_meta()` - 加载或计算所有 horizon 步骤的聚合统计
- `_get_lerobot_relative_horizon_stats_meta()` - 加载或计算每个 horizon 步骤的独立统计
- `_calculate_relative_stats_for_key()` - 对特定动作键计算相对统计
- `_calculate_relative_stats_for_key_per_horizon()` - 对特定动作键按 horizon 分别计算统计

计算公式：`relative_action = action[t + delta] - state[t]`

#### 数据加载核心方法

- `_get_trajectories()` - 从 `episodes.jsonl` 提取轨迹 ID 和长度
- `_get_all_steps()` - 生成 `(trajectory_id, base_index)` 对的列表，支持步骤过滤和坏轨迹丢弃
- `_get_metadata()` - 将 LeRobot 原始元数据转换为统一的 `DatasetMetadata` 格式，处理视频分辨率、状态/动作维度等信息
- `_load_trajectory_data()` - 从 parquet 文件加载单条轨迹数据

#### `__getitem__` 方法

这是 PyTorch Dataset 的核心接口，完成以下步骤：
1. 根据索引获取 `(trajectory_id, base_index)`
2. 加载对应的 parquet 数据
3. 按 `delta_indices` 切片出所需的时间步
4. 加载视频帧（支持 ffmpeg 后端）
5. 加载语言标注
6. 如果启用 `relative_action`，将动作转换为相对值
7. 应用变换管线

### 5. `LeRobotMixtureDataset` 类

用于混合多个数据集进行训练。核心特性：

- **加权采样** - 支持按数据集设定采样权重
- **自动元数据合并** - 使用 YAML 配置文件定义混合策略
- **分布式支持** - 支持多 GPU 训练场景下的数据分片

```python
class LeRobotMixtureDataset(Dataset):
    @classmethod
    def from_yaml(cls, yaml_path, ...) -> "LeRobotMixtureDataset":
        # 从 YAML 配置文件创建混合数据集
```

## 核心类/函数表格

| 名称 | 类型 | 说明 |
|------|------|------|
| `calculate_dataset_statistics` | 函数 | 计算 parquet 数据的统计值 |
| `ModalityConfig` | Pydantic 模型 | 模态采样配置（delta_indices、modality_keys） |
| `LeRobotSingleDataset` | PyTorch Dataset | 单 LeRobot 数据集，支持视频/状态/动作/语言多模态加载 |
| `LeRobotMixtureDataset` | PyTorch Dataset | 混合数据集，支持加权采样和 YAML 配置 |

## 与其他模块的关系

- **依赖 `schema`** - 使用 `DatasetMetadata`、`EmbodimentTag`、`LeRobotModalityMetadata` 等数据结构
- **依赖 `macro.py`** - 使用 `LE_ROBOT_EPISODE_FILENAME` 等常量
- **依赖 `transform`** - 使用 `ComposedModalityTransform` 作为变换管线
- **依赖 `conversion`** - 使用 `load_initial_actions` 加载初始动作
- **被 `lerobot_sharded.py` 继承** - 分片数据集基于此类扩展

## 总结

`lerobot.py` 是 DreamZero 数据加载的核心引擎。它设计了一套灵活的模态配置系统（通过 `ModalityConfig` 和 `delta_indices`），使得不同模态可以按不同的时间窗口采样数据。相对动作统计功能是 DreamZero 相对于标准 LeRobot 的重要扩展，支持更精确的动作归一化。混合数据集机制则为跨数据集训练提供了便利。
