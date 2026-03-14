# base_48_wan_fine_aug_relative.yaml -- 基础数据管道配置

## 文件概述

`base_48_wan_fine_aug_relative.yaml` 是 DreamZero 数据配置的**基础文件**，被所有具体数据集配置（如 `droid_relative.yaml`、`agibot_relative.yaml`、`yam_relative.yaml`）继承。它定义了：

1. **维度信息**：视频帧数、动作时域、图像分辨率等。
2. **视频数据增强管道**：裁剪、缩放、颜色抖动、灰度化、归一化等变换。
3. **多数据集模态配置**：为 DROID、AGIBot、YAM 三种数据集分别定义视频/状态/动作/语言的模态结构。
4. **数据变换链**：每种数据集的完整预处理流水线。
5. **元数据版本与 FPS**。

## 关键配置项解析

### 1. 维度信息

```yaml
num_frames: 49
action_horizon: 48
state_horizon: 1
image_resolution_width: 480
image_resolution_height: 256
```

- 每个训练样本包含 49 帧视频，对应 48 步动作预测。
- 视频分辨率为 480x256（宽高比约 1.875:1）。
- 单帧模式下使用 256x256 正方形分辨率。

### 2. 视频数据增强（YAML Anchors）

配置使用 YAML 锚点（`&anchor`/`*anchor`）定义可复用的增强组件：

| 增强组件 | 类 | 关键参数 |
|---------|------|---------|
| `totensor_cfg` | `VideoToTensor` | 将视频帧转为张量 |
| `crop_cfg` | `VideoCrop` | 随机裁剪，scale=0.95 |
| `resize_cfg` | `VideoResize` | 缩放到 480x256 |
| `color_jitter_cfg` | `VideoColorJitter` | 亮度 0.3，对比度 0.4，饱和度 0.5，色调 0.08 |
| `random_grayscale_cfg` | `VideoRandomGrayscale` | 10% 概率灰度化 |
| `random_posterize_cfg` | `VideoRandomPosterize` | 10% 概率色调量化（4 bit） |
| `normalize_cfg` | `VideoNormalize` | 归一化到 [-1, 1] |
| `to_numpy_cfg` | `VideoToNumpy` | 转为 NumPy 数组 |

### 3. DROID 数据集模态配置

```yaml
modality_config_oxe_droid:
  video:
    delta_indices: [0, 1, ..., 24]    # 25 帧视频
    modality_keys: [exterior_image_1_left, exterior_image_2_left, wrist_image_left]  # 3 视角
  state:
    delta_indices: [0]                  # 1 步状态
    modality_keys: [joint_position, gripper_position]
  action:
    delta_indices: [0, 1, ..., 23]      # 24 步动作
    modality_keys: [joint_position, gripper_position]
  language:
    modality_keys: [language_instruction, language_instruction_2, language_instruction_3]
```

DROID 使用 3 个摄像头视角、关节+夹爪的状态/动作空间，以及最多 3 条语言指令。

### 4. AGIBot 数据集模态配置

AGIBot 具有更丰富的模态：
- **视频**：3 视角（top_head、hand_left、hand_right）
- **状态**：6 组（左右臂关节、左右手效应器、头部、腰部）
- **动作**：7 组（在状态基础上增加 robot_velocity）
- **语言**：`action_text`

### 5. YAM 数据集模态配置

YAM 的模态：
- **视频**：3 视角（top_camera、left_camera、right_camera）
- **状态/动作**：各 4 组（左右臂关节+夹爪）
- **语言**：`annotation.task`

### 6. 数据变换链

每种数据集的变换链遵循相同结构：
1. 视频变换：ToTensor → Crop → Resize → ColorJitter → ToNumpy
2. 状态变换：ToTensor → q99 归一化
3. 动作变换：ToTensor → q99 归一化
4. 拼接变换：将多视角/多模态数据拼接
5. 模型特定变换：`${model_specific_transform}`

### 7. 元数据版本与 FPS

```yaml
metadata_versions:
  oxe_droid: '0221'
  agibot: '0221'
  yam: '0221'

fps:
  yam: 30
```

所有数据集使用 `0221` 版本的归一化元数据，YAM 指定了 30 FPS。

## 配置参数表格

| 参数名 | 值 | 说明 |
|--------|------|------|
| `num_frames` | `49` | 每样本视频帧数 |
| `action_horizon` | `48` | 动作预测时域长度 |
| `state_horizon` | `1` | 状态观测时域长度 |
| `image_resolution_width` | `480` | 视频帧宽度 |
| `image_resolution_height` | `256` | 视频帧高度 |
| `image_resolution_width_single_frame` | `256` | 单帧宽度 |
| `image_resolution_height_single_frame` | `256` | 单帧高度 |
| `use_global_metadata` | `false` | 使用局部归一化统计 |
| `crop.scale` | `0.95` | 随机裁剪缩放因子 |
| `color_jitter.brightness` | `0.3` | 亮度抖动范围 |
| `color_jitter.contrast` | `0.4` | 对比度抖动范围 |
| `color_jitter.saturation` | `0.5` | 饱和度抖动范围 |
| `color_jitter.hue` | `0.08` | 色调抖动范围 |
| `random_grayscale.p` | `0.1` | 灰度化概率 |
| `normalize.mean/std` | `[0.5, 0.5, 0.5]` | 归一化参数 |

## 与其他配置/模块的关系

- **被继承**：被 `droid_relative.yaml`、`agibot_relative.yaml`、`yam_relative.yaml` 通过 Hydra defaults 继承。
- **引用 model_specific_transform**：在每个数据集的变换链末尾插入 `${model_specific_transform}`，该变量由 `transform/dreamzero_cotrain.yaml` 定义。
- **提供全局维度参数**：`num_frames`、`action_horizon`、`state_horizon` 等参数被 action_head 和 transform 配置引用。
- **归一化模式**：所有状态和动作使用 `q99`（99 分位数）归一化模式。

## 总结

`base_48_wan_fine_aug_relative.yaml` 是 DreamZero 数据管道的核心基础配置。它以 49 帧视频 + 48 步动作的时域结构为基础，定义了适用于视频生成任务的数据增强策略（裁剪、颜色抖动、灰度化等），并为 DROID、AGIBot、YAM 三种机器人平台提供了完整的模态定义和变换管道。所有具体数据集配置都通过继承此文件来复用这些基础设施，仅需覆盖数据路径和相对动作键等特定设置。
