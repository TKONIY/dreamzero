# droid_relative.yaml -- DROID 数据集配置

## 文件概述

`droid_relative.yaml` 定义了使用 **DROID（Distributed Robot Interaction Dataset）** 数据集进行 DreamZero 训练的配置。DROID 是一个大规模多机器人操作数据集，此配置文件继承基础数据配置，并针对 DROID 数据集的特性进行了定制，支持相对动作（relative action）模式和分片数据加载。

## 关键配置项解析

### 1. 继承基础配置

```yaml
defaults:
  - dreamzero/base_48_wan_fine_aug_relative
  - _self_
```

继承 `base_48_wan_fine_aug_relative.yaml`（48 步动作时域 + WAN 微调 + 数据增强 + 相对动作），当前文件的设置会覆盖基础配置。

### 2. 相对动作设置

```yaml
relative_action: true
relative_action_per_horizon: false
relative_action_keys:
  - joint_position
```

- `relative_action: true`：使用相对动作表示（相对于当前帧的增量），而非绝对位置。
- `relative_action_per_horizon: false`：相对动作以初始帧为基准计算，而非逐步累积。
- `relative_action_keys`：仅对 `joint_position` 使用相对动作。

### 3. 分片数据加载

```yaml
dataset_shard_sampling_rate: 0.1
mixture_dataset_cls: groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotMixtureDataset.from_mixture_spec
single_dataset_cls: groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotSubLangSingleActionChunkDatasetDROID
```

使用分片（Sharded）数据加载策略，每次只加载 10% 的数据分片，适合处理超大规模数据集。

### 4. 数据集混合配置

```yaml
train_dataset:
  mixture_spec:
    - dataset_path:
        oxe_droid:
          - ${droid_data_root}
      dataset_weight: 1.0
      distribute_weights: true
```

配置 DROID 数据的混合规格，权重为 1.0（单数据集训练时）。`droid_data_root` 需要用户在命令行指定。

## 配置参数表格

| 参数名 | 值 | 说明 |
|--------|------|------|
| `max_state_dim` | `64` | 最大状态维度 |
| `use_global_metadata` | `false` | 不使用全局归一化统计 |
| `relative_action` | `true` | 使用相对动作 |
| `relative_action_per_horizon` | `false` | 不逐步累积相对动作 |
| `relative_action_keys` | `[joint_position]` | 使用相对动作的键 |
| `max_chunk_size` | `5` | 最大分块大小 |
| `dataset_shard_sampling_rate` | `0.1` | 分片采样率（10%） |
| `droid_data_root` | `???`（必填） | DROID 数据集根路径 |
| `video_backend` | `decord` | 视频解码后端 |
| `seed` | `42` | 数据采样随机种子 |

## 与其他配置/模块的关系

- **继承自**：`data/dreamzero/base_48_wan_fine_aug_relative.yaml`（基础数据管道配置）。
- **被引用**：可通过 `conf.yaml` 的 `data: dreamzero/droid_relative` 引入。
- **数据集类型**：使用 `oxe_droid` 体型标签，对应 `base.yaml` 中的投影器索引 17。
- **数据变换**：通过 `all_transforms: ${transforms}` 引用基础配置中定义的 DROID 变换管道。

## 总结

`droid_relative.yaml` 是针对 DROID 数据集的单数据集训练配置。它采用相对动作表示和分片数据加载策略，适合在大规模 DROID 数据上进行高效训练。用户只需通过命令行设置 `droid_data_root` 即可启动训练。
