# 分片 LeRobot 数据集 (`groot/vla/data/dataset/lerobot_sharded.py`)

## 文件概述

`lerobot_sharded.py` 实现了支持分片加载的 LeRobot 数据集，专为大规模数据集的高效训练而设计。它通过将数据集划分为多个 shard（分片），实现了按需加载和缓存机制，显著降低了内存占用，并支持分布式训练场景下的数据分配。

## 关键代码解析

### 1. `ShardedLeRobotSingleDataset` 类

继承自 `LeRobotSingleDataset`，核心改进是将数据按固定步数划分为多个 shard：

```python
class ShardedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, *args, num_steps_per_shard: int = int(1e4), **kwargs):
        super().__init__(*args, **kwargs)
        self.num_steps_per_shard = num_steps_per_shard
        self.sharded_trajectories, self.shard_lengths = self.generate_shards()
        self.frames_to_load = self.get_all_frames_to_load()
```

#### 分片生成逻辑

`generate_shards()` 方法将轨迹按 `num_steps_per_shard` 分组：

- 遍历所有轨迹，累计步数
- 当累计步数超过 `num_steps_per_shard` 时，创建新的 shard
- 每个 shard 包含若干完整轨迹

#### 缓存机制

使用 `ThreadPoolExecutor` 实现异步预加载：

```python
self._executor = ThreadPoolExecutor(max_workers=1)
self._cache_job: Future | None = None
```

- `cached_shard` - 当前缓存的 shard 数据（numpy 数组）
- `cached_df` - 当前缓存的 DataFrame
- `frame_indices_map` - 视频帧索引映射

当切换到新的 shard 时，系统会异步预加载下一个 shard 的数据。

#### Shard 加载流程

```python
def load_shard(self, shard_idx: int):
    # 1. 获取该 shard 包含的轨迹
    # 2. 加载所有轨迹的 parquet 数据
    # 3. 预加载视频帧到内存
    # 4. 更新 shard 起始索引映射
```

### 2. `ShardedLeRobotMixtureDataset` 类

继承自 `IterableDataset`，实现了分片混合数据集，核心特性：

- **基于 `IterableDataset`** - 适合大规模流式数据处理
- **分布式数据分配** - 根据 `torch.distributed` 的 rank 和 world_size 分配 shard
- **Worker 级别分片** - 支持 `DataLoader` 的多 worker 场景

```python
def __iter__(self):
    # 1. 获取当前 worker 和 rank 信息
    # 2. 根据 rank 分配 shard 子集
    # 3. 在 worker 之间进一步划分 shard
    # 4. 按顺序加载和遍历每个 shard
```

#### 采样策略

混合数据集中的各子数据集按权重采样：

```python
def _adjust_shard_weights(self):
    # 根据数据集权重和 shard 大小调整采样概率
```

### 3. 视频帧预加载

`get_all_frames_to_load()` 方法预计算每条轨迹需要加载的视频帧列表，避免重复 I/O：

```python
def get_all_frames_to_load(self):
    # 基于 delta_indices 计算每条轨迹的帧范围
    # 返回 {轨迹ID: {视频键: [帧索引列表]}}
```

## 核心类/函数表格

| 名称 | 类型 | 说明 |
|------|------|------|
| `ShardedLeRobotSingleDataset` | Dataset | 分片单数据集，支持 shard 缓存和异步预加载 |
| `ShardedLeRobotMixtureDataset` | IterableDataset | 分片混合数据集，支持分布式训练和多 worker 加载 |
| `generate_shards` | 方法 | 将轨迹按步数阈值划分为多个 shard |
| `load_shard` | 方法 | 加载指定 shard 的所有数据到内存 |
| `get_all_frames_to_load` | 方法 | 预计算每条轨迹需要的视频帧索引 |

## 与其他模块的关系

- **继承 `lerobot.py`** - `ShardedLeRobotSingleDataset` 继承 `LeRobotSingleDataset`，复用元数据加载和数据处理逻辑
- **依赖 `common.utils`** - 使用 `get_frames_by_timestamps` 加载视频帧
- **依赖 PyTorch 分布式** - 使用 `torch.distributed` 和 `get_worker_info()` 实现数据分配

## 总结

`lerobot_sharded.py` 通过分片和缓存机制解决了大规模数据集的内存和 I/O 瓶颈问题。它的设计考虑了分布式训练中的 rank 分配、DataLoader 的多 worker 协作以及异步数据预加载，是 DreamZero 支持大规模训练的关键组件。每个 shard 默认包含 10000 步数据，在内存占用和加载效率之间取得平衡。
