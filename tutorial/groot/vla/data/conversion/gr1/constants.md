# GR1 数据常量定义 (`groot/vla/data/conversion/gr1/constants.py`)

## 文件概述

`constants.py` 定义了 GR1 机器人数据转换过程中使用的文件名常量和 embodiment 标签映射关系。这些常量服务于原始数据到可训练 LeRobot 格式的转换流程。

## 关键代码解析

### 1. 原始数据文件名

```python
RAW_HDF5_FILENAME = "state_action.hdf5"       # 原始状态/动作数据
RAW_METADATA_FILENAME = "metadata.json"         # 原始元数据
RAW_VIDEO_FILENAME = "ego_view.mp4"             # 原始第一人称视频
RAW_ANNOTATION_FILENAME = "annotation.json"     # 原始标注数据
```

这些是 GR1 机器人数据采集后的原始文件命名约定。

### 2. 可训练格式文件名

```python
TRAINABLE_HDF5_FILENAME = "dataset.hdf5"        # 处理后的 HDF5 数据
TRAINABLE_METADATA_FILENAME = "metadata.json"    # 处理后的元数据
```

### 3. 控制频率

```python
RAW_DATA_CONTROL_FREQUENCY = 20  # 20Hz 控制频率
```

GR1 数据采集的控制频率为 20Hz。

### 4. 初始动作文件

```python
INITIAL_ACTIONS_FILENAME = "initial_actions.npz"
```

存储每条轨迹第一步动作的文件，用于相对动作计算。

### 5. Embodiment 标签映射

```python
EMBODIMENT_TAG_TO_ANNOTATED_VERSION = {
    EmbodimentTag.REAL_GR1_ARMS_ONLY: EmbodimentTag.REAL_GR1_ARMS_ONLY_ANNOTATED,
    EmbodimentTag.REAL_GR1_ARMS_WAIST: EmbodimentTag.REAL_GR1_ARMS_WAIST_ANNOTATED,
    EmbodimentTag.ROBOCASA_GR1_ARMS_ONLY_FOURIER_HANDS: EmbodimentTag.REAL_GR1_ARMS_ONLY_ANNOTATED,
    EmbodimentTag.ROBOCASA_GR1_ARMS_WAIST_FOURIER_HANDS: EmbodimentTag.REAL_GR1_ARMS_WAIST_ANNOTATED,
}
```

将基础 embodiment 标签映射到其带标注的版本。注意 RoboCasa 模拟数据的 Fourier 手版本映射到对应的真实 GR1 标注版本。

### 6. 其他常量

```python
EPISODE_LENGTH_FILENAME = "episode_length.json"            # 分段数据的剧集长度
PROCESSED_VIDEO_FILENAME = "ego_view_pad_res224_freq20.mp4" # 处理后的视频
```

## 核心类/函数表格

| 名称 | 类型 | 说明 |
|------|------|------|
| `RAW_HDF5_FILENAME` | 常量 | 原始 HDF5 数据文件名 |
| `TRAINABLE_HDF5_FILENAME` | 常量 | 可训练 HDF5 数据文件名 |
| `RAW_DATA_CONTROL_FREQUENCY` | 常量 | 数据采集控制频率 (20Hz) |
| `INITIAL_ACTIONS_FILENAME` | 常量 | 初始动作文件名 |
| `EMBODIMENT_TAG_TO_ANNOTATED_VERSION` | 字典 | 基础标签到标注版本的映射 |

## 与其他模块的关系

- **依赖 `schema`** - 使用 `EmbodimentTag` 枚举
- **被 `get_initial_actions.py` 使用** - 引用 `TRAINABLE_HDF5_FILENAME` 和 `INITIAL_ACTIONS_FILENAME`
- **被转换脚本使用** - 数据格式转换流程中引用这些常量

## 总结

`constants.py` 集中定义了 GR1 数据转换过程中的所有文件名约定和标签映射，使得数据转换流程的配置统一可控。控制频率 20Hz 是 GR1 平台的硬件特性参数，直接影响视频帧采样和动作序列的时间分辨率。
