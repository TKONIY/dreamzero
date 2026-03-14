# yam_relative.yaml -- YAM 数据集配置

## 文件概述

`yam_relative.yaml` 定义了使用 **YAM** 机器人数据集进行 DreamZero 训练的配置。YAM 是一种双臂机器人平台，每臂具有关节位置和夹爪位置两组控制维度。此配置针对 YAM 的状态/动作空间特性进行了相对动作设置。

## 关键配置项解析

### 1. 继承基础配置

```yaml
defaults:
  - dreamzero/base_48_wan_fine_aug_relative
  - _self_
```

### 2. 双臂关节与夹爪的相对动作

```yaml
relative_action_keys:
  - left_joint_pos
  - left_gripper_pos
  - right_joint_pos
  - right_gripper_pos
```

YAM 的相对动作包含 4 组键：左臂关节位置、左夹爪位置、右臂关节位置和右夹爪位置。

### 3. 数据集路径

```yaml
yam_data_root: ???

train_dataset:
  mixture_spec:
    - dataset_path:
        yam:
          - ${yam_data_root}
```

使用 `yam` 体型标签，用户需通过命令行提供 `yam_data_root`。

## 配置参数表格

| 参数名 | 值 | 说明 |
|--------|------|------|
| `max_state_dim` | `64` | 最大状态维度 |
| `use_global_metadata` | `false` | 不使用全局归一化统计 |
| `relative_action` | `true` | 使用相对动作 |
| `relative_action_per_horizon` | `false` | 不逐步累积相对动作 |
| `relative_action_keys` | 4 组（左右臂关节+夹爪） | 使用相对动作的键 |
| `max_chunk_size` | `5` | 最大分块大小 |
| `dataset_shard_sampling_rate` | `0.1` | 分片采样率 |
| `yam_data_root` | `???`（必填） | YAM 数据集根路径 |

## 与其他配置/模块的关系

- **继承自**：`data/dreamzero/base_48_wan_fine_aug_relative.yaml`。
- **体型标签**：使用 `yam`，对应 `base.yaml` 中的投影器索引 32。
- **数据变换**：引用基础配置中的 `transform_yam`，包含视频增强、状态/动作 q99 归一化等管道。
- **模态配置**：引用 `modality_config_yam`，定义了 3 视角视频（top_camera、left_camera、right_camera）、4 组状态、4 组动作和语言指令（annotation.task）。
- **FPS 设置**：基础配置中为 YAM 指定了 `fps: 30`。

## 总结

`yam_relative.yaml` 为 YAM 双臂机器人提供了训练数据配置。其状态/动作空间涵盖左右臂的关节和夹爪位置，总共 14 维（与 AGIBot 的 30+ 维相比更为紧凑）。配置结构与 DROID、AGIBot 保持一致，支持灵活的数据集切换。
