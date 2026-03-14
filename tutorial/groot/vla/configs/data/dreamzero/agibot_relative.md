# agibot_relative.yaml -- AGIBot 数据集配置

## 文件概述

`agibot_relative.yaml` 定义了使用 **AGIBot** 机器人数据集进行 DreamZero 训练的配置。AGIBot 是一种人形双臂机器人，具有丰富的状态和动作空间（包括双臂、双手效应器、头部、腰部和底盘速度），此配置针对其多关节特性进行了专门的相对动作设置。

## 关键配置项解析

### 1. 继承基础配置

```yaml
defaults:
  - dreamzero/base_48_wan_fine_aug_relative
  - _self_
```

### 2. 多关节相对动作

```yaml
relative_action_keys:
  - left_arm_joint_position
  - right_arm_joint_position
  - left_effector_position
  - right_effector_position
  - head_position
  - waist_position
```

与 DROID 配置相比，AGIBot 的相对动作键列表更丰富，涵盖了人形机器人的所有主要关节组：左臂、右臂、左手效应器、右手效应器、头部和腰部。

### 3. 数据集路径

```yaml
agibot_data_root: ???

train_dataset:
  mixture_spec:
    - dataset_path:
        agibot:
          - ${agibot_data_root}
```

使用 `agibot` 体型标签，用户需通过命令行提供 `agibot_data_root`。

## 配置参数表格

| 参数名 | 值 | 说明 |
|--------|------|------|
| `max_state_dim` | `64` | 最大状态维度 |
| `use_global_metadata` | `false` | 不使用全局归一化统计 |
| `relative_action` | `true` | 使用相对动作 |
| `relative_action_per_horizon` | `false` | 不逐步累积相对动作 |
| `relative_action_keys` | 6 个关节组 | 使用相对动作的关节键 |
| `max_chunk_size` | `5` | 最大分块大小 |
| `dataset_shard_sampling_rate` | `0.1` | 分片采样率 |
| `agibot_data_root` | `???`（必填） | AGIBot 数据集根路径 |

## 与其他配置/模块的关系

- **继承自**：`data/dreamzero/base_48_wan_fine_aug_relative.yaml`。
- **体型标签**：使用 `agibot`，对应 `base.yaml` 中的投影器索引 26。
- **数据变换**：引用基础配置中的 `transform_agibot`，包含视频增强、状态/动作归一化等完整管道。
- **模态配置**：引用 `modality_config_agibot`，定义了 3 视角视频（top_head、hand_left、hand_right）、6 组状态、7 组动作和语言指令。

## 总结

`agibot_relative.yaml` 为 AGIBot 人形机器人提供了完整的训练数据配置。其相对动作键覆盖了双臂、双手、头部和腰部共 6 个关节组，充分利用了 AGIBot 丰富的运动自由度。配置结构与 DROID 配置保持一致，便于在不同机器人平台间切换。
