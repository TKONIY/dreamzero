# LeRobot 数据模式定义 (`groot/vla/data/schema/lerobot.py`)

## 文件概述

`lerobot.py` 是数据模式定义的核心文件，基于 Pydantic 定义了 DreamZero 数据管线中使用的所有数据结构。文件分为三层：LeRobot 原始模式、通用模式（旋转类型）和简化后的数据集模式。所有结构都支持 JSON 序列化/反序列化，用于元数据的持久化存储。

## 关键代码解析

### 1. 旋转类型枚举

```python
class RotationType(Enum):
    AXIS_ANGLE = "axis_angle"
    QUATERNION = "quaternion"
    ROTATION_6D = "rotation_6d"
    MATRIX = "matrix"
    EULER_ANGLES_RPY = "euler_angles_rpy"
    # ... 以及其他欧拉角顺序
```

定义了机器人动作/状态中旋转信息的所有可能表示方式。这对于 `state_action.py` 中的旋转转换至关重要。

### 2. LeRobot 模态字段

#### `LeRobotModalityField`

```python
class LeRobotModalityField(BaseModel):
    original_key: Optional[str] = None  # LeRobot 数据集中的原始键名
```

所有模态字段的基类，记录原始数据集中的键名映射。

#### `LeRobotStateActionMetadata`

```python
class LeRobotStateActionMetadata(LeRobotModalityField):
    start: int           # 在拼接向量中的起始索引
    end: int             # 在拼接向量中的结束索引
    rotation_type: Optional[RotationType]  # 旋转类型（如果适用）
    absolute: bool       # 是否为绝对值
    dtype: str           # 数据类型
    range: Optional[tuple[float, float]]  # 值域范围
    original_key: str    # LeRobot 原始键名
```

这是理解数据切片逻辑的关键类。LeRobot 将多个状态/动作分量拼接为一个向量（如 `observation.state`），此类通过 `start` 和 `end` 指定每个分量在拼接向量中的位置。

#### `LeRobotStateMetadata` 和 `LeRobotActionMetadata`

分别继承 `LeRobotStateActionMetadata`，设置了不同的默认 `original_key`：

```python
class LeRobotStateMetadata(LeRobotStateActionMetadata):
    original_key: str = "observation.state"  # 状态的默认原始键

class LeRobotActionMetadata(LeRobotStateActionMetadata):
    original_key: str = "action"             # 动作的默认原始键
```

### 3. `LeRobotModalityMetadata`

```python
class LeRobotModalityMetadata(BaseModel):
    state: dict[str, LeRobotStateMetadata]
    action: dict[str, LeRobotActionMetadata]
    video: dict[str, LeRobotModalityField]
    annotation: Optional[dict[str, LeRobotModalityField]]
```

完整的模态元数据定义。`model_validator` 自动为未设置 `original_key` 的字段生成默认值。

重要方法：

- `get_key_meta(key)` - 通过 `"modality.subkey"` 格式获取元数据，例如 `get_key_meta("state.joint_position")`
- `get_original_key(key)` - 获取原始 LeRobot 键名

### 4. 简化数据集模式

这些类是 LeRobot 原始模式的简化版本，用于内部数据处理：

#### `DatasetStatisticalValues`

```python
class DatasetStatisticalValues(BaseModel):
    max: np.ndarray     # 最大值
    min: np.ndarray     # 最小值
    mean: np.ndarray    # 均值
    std: np.ndarray     # 标准差
    q01: np.ndarray     # 1% 分位数
    q99: np.ndarray     # 99% 分位数
```

使用自定义的 `field_serializer` 和 `field_validator` 实现 numpy 数组与 JSON 列表之间的自动转换：

```python
@field_serializer("*", when_used="json")
def serialize_ndarray(self, v: np.ndarray) -> list:
    return v.tolist()

@field_validator("*", mode="before")
def validate_ndarray(cls, v) -> np.ndarray:
    return np.array(v)
```

#### `DatasetMetadata`

```python
class DatasetMetadata(BaseModel):
    statistics: DatasetStatistics        # 数据集统计信息
    modalities: DatasetModalities        # 模态元数据
    embodiment_tag: EmbodimentTag        # 机器人构型标签
```

这是 DreamZero 内部使用的最终元数据格式，整合了统计信息、模态定义和构型标识。

#### `StateActionMetadata`

```python
class StateActionMetadata(BaseModel):
    absolute: bool            # 是否为绝对值
    rotation_type: Optional[RotationType]  # 旋转类型
    shape: tuple[int, ...]    # 形状
    continuous: bool          # 是否为连续值
```

简化后的状态/动作元数据，用于变换管线中的维度计算和归一化策略选择。

## 核心类/函数表格

| 名称 | 类型 | 说明 |
|------|------|------|
| `RotationType` | Enum | 旋转表示类型枚举 |
| `LeRobotModalityField` | Pydantic 模型 | 模态字段基类 |
| `LeRobotStateActionMetadata` | Pydantic 模型 | 状态/动作元数据，包含索引切片信息 |
| `LeRobotModalityMetadata` | Pydantic 模型 | 完整模态元数据，包含 state/action/video/annotation |
| `DatasetStatisticalValues` | Pydantic 模型 | 统计值（均值/标准差/分位数等），支持 numpy 序列化 |
| `DatasetStatistics` | Pydantic 模型 | state 和 action 的统计信息集合 |
| `DatasetMetadata` | Pydantic 模型 | 最终数据集元数据，包含统计、模态和构型信息 |
| `StateActionMetadata` | Pydantic 模型 | 简化的状态/动作元数据 |
| `VideoMetadata` | Pydantic 模型 | 视频元数据（分辨率、通道数、帧率） |

## 与其他模块的关系

- **被 `dataset/lerobot.py` 使用** - 加载和构建 `DatasetMetadata`、`LeRobotModalityMetadata` 等
- **被 `transform/base.py` 使用** - `ModalityTransform` 依赖 `DatasetMetadata` 获取归一化参数
- **被 `transform/state_action.py` 使用** - 使用 `RotationType` 和 `StateActionMetadata` 控制旋转转换和归一化
- **被 `transform/concat.py` 使用** - 使用 `StateActionMetadata` 获取维度信息

## 总结

`schema/lerobot.py` 是 DreamZero 数据类型系统的基石。它通过两层抽象（LeRobot 原始模式和简化数据集模式）实现了从外部数据格式到内部处理格式的桥接。基于 Pydantic 的设计提供了自动验证、序列化和文档生成能力。其中 `start`/`end` 索引切片机制是理解数据从 LeRobot 拼接向量到 DreamZero 分离模态的关键。
