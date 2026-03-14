# 状态/动作变换 (`groot/vla/data/transform/state_action.py`)

## 文件概述

`state_action.py` 是变换模块中最复杂的文件，实现了状态和动作数据的完整预处理管线。主要包含旋转表示转换、多种归一化策略、数据类型转换、噪声扰动和 Dropout 等功能。该文件约 900 行代码，定义了 8 个类。

## 关键代码解析

### 1. `RotationTransform` 类

```python
class RotationTransform:
    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]
```

基于 pytorch3d 实现旋转表示之间的转换。设计采用"通过矩阵中间表示"的策略：

```
源表示 -> 旋转矩阵 -> 目标表示
```

初始化时构建正向和逆向函数链：

```python
def __init__(self, from_rep="axis_angle", to_rep="rotation_6d"):
    if from_rep != "matrix":
        forward_funcs.append(pt.{from_rep}_to_matrix)
        inverse_funcs.append(pt.matrix_to_{from_rep})
    if to_rep != "matrix":
        forward_funcs.append(pt.matrix_to_{to_rep})
        inverse_funcs.append(pt.{to_rep}_to_matrix)
```

对欧拉角，支持多种旋转顺序（RPY、RYP 等），通过 `functools.partial` 绑定 convention 参数。

### 2. `Normalizer` 类

支持 5 种归一化模式：

| 模式 | 公式 | 输出范围 |
|------|------|----------|
| `q99` | `2 * (x - q01) / (q99 - q01) - 1` | `[-1, 1]` |
| `mean_std` | `(x - mean) / std` | 无固定范围 |
| `min_max` | `2 * (x - min) / (max - min) - 1` | `[-1, 1]` |
| `scale` | `x / max(abs(min), abs(max))` | `[-1, 1]` |
| `binary` | `x > 0.5 ? 1 : 0` | `{0, 1}` |

每种模式都处理了除零异常（如 `q01 == q99`、`std == 0`）：

```python
mask = q01 != q99
normalized[..., mask] = (x[..., mask] - q01[..., mask]) / (q99[..., mask] - q01[..., mask])
normalized[..., ~mask] = x[..., ~mask]  # 保持原值
```

`inverse` 方法实现反归一化，用于推理时将模型输出还原为真实动作值。

### 3. `PerHorizonNormalizer` 类

与 `Normalizer` 类似，但为每个 horizon 步骤使用不同的统计参数：

```python
class PerHorizonNormalizer:
    # statistics: {"q01": tensor(horizon_len, action_dim), ...}
```

关键处理：自动检测输入形状并适配：

```python
if len(x.shape) == 2:
    total_len = x.shape[0]
    if total_len > stats_horizon_len and total_len % stats_horizon_len == 0:
        # 多 chunk 场景: reshape 为 (num_chunks, horizon_len, action_dim)
        x = x.view(num_chunks, stats_horizon_len, action_dim)
```

### 4. `StateActionToTensor` 类

将 numpy 数组转换为 PyTorch 张量：

```python
class StateActionToTensor(InvertibleModalityTransform):
    def apply(self, data):
        for key in self.apply_to:
            data[key] = torch.from_numpy(data[key])
    def unapply(self, data):
        for key in self.apply_to:
            data[key] = data[key].numpy()
```

支持自定义输入/输出数据类型映射。

### 5. `StateActionTransform` 类（核心）

整合旋转转换和归一化的主变换类：

```python
class StateActionTransform(InvertibleModalityTransform):
    apply_to: list[str]                          # 目标键
    normalization_modes: dict[str, str]          # 归一化模式
    target_rotations: dict[str, str]             # 目标旋转表示
    normalization_statistics: dict[str, dict]    # 归一化统计数据
    modality_metadata: dict[str, StateActionMetadata]  # 模态元数据
```

#### `set_metadata` 方法

初始化旋转转换器和归一化器的完整逻辑：

1. 从 `DatasetMetadata` 提取各键的模态配置
2. 从 `DatasetStatistics` 提取统计信息
3. 初始化 `RotationTransform`（如果需要旋转转换）
4. 初始化 `Normalizer`（根据指定的归一化模式）

对于绝对旋转+旋转转换的特殊情况，使用预定义的统计范围：

```python
_DEFAULT_MIN_MAX_STATISTICS = {
    "rotation_6d": {"min": [-1]*6, "max": [1]*6},
    "quaternion": {"min": [-1]*4, "max": [1]*4},
    "axis_angle": {"min": [-pi]*3, "max": [pi]*3},
}
```

#### `apply` 方法

按顺序执行：旋转转换 -> 归一化：

```python
def apply(self, data):
    for key in self.apply_to:
        state = data[key]
        if key in self._rotation_transformers:
            state = self._rotation_transformers[key].forward(state)
        if key in self._normalizers:
            state = self._normalizers[key].forward(state)
        data[key] = state
```

#### `unapply` 方法

逆序执行：反归一化 -> 逆旋转：

```python
def unapply(self, data):
    for key in self.apply_to:
        if key in self._normalizers:
            state = self._normalizers[key].inverse(state)
        if key in self._rotation_transformers:
            state = self._rotation_transformers[key].inverse(state)
```

### 6. `PerHorizonActionTransform` 类

使用 per-horizon 统计数据进行动作归一化：

```python
class PerHorizonActionTransform(InvertibleModalityTransform):
    normalization_modes: dict[str, str]
    per_horizon_statistics: dict[str, dict[str, list]]
```

通过 `set_per_horizon_statistics` 方法接收统计数据并初始化 `PerHorizonNormalizer` 实例。

### 7. `StateActionPerturbation` 类

训练时为状态/动作添加高斯噪声：

```python
class StateActionPerturbation(ModalityTransform):
    std: float  # 噪声标准差
    def apply(self, data):
        if not self.training: return data  # 评估时不添加噪声
        noise = torch.randn_like(state) * self.std
        state = torch.clamp(state + noise, min_val, max_val)
```

### 8. `StateActionDropout` 类

训练时以一定概率将状态/动作置零：

```python
class StateActionDropout(ModalityTransform):
    dropout_prob: float
    def apply(self, data):
        if random.random() < self.dropout_prob:
            state = torch.zeros_like(state)
```

### 9. `StateActionSinCosTransform` 类

将角度值转换为 sin-cos 编码：

```python
class StateActionSinCosTransform(ModalityTransform):
    def apply(self, data):
        data[key] = torch.cat([torch.sin(state), torch.cos(state)], dim=-1)
```

维度从 D 变为 2D。

## 核心类/函数表格

| 名称 | 类型 | 说明 |
|------|------|------|
| `RotationTransform` | 工具类 | 旋转表示转换（轴角/四元数/6D/矩阵/欧拉角） |
| `Normalizer` | 工具类 | 支持 5 种模式的归一化/反归一化 |
| `PerHorizonNormalizer` | 工具类 | 逐 horizon 步骤的归一化器 |
| `StateActionToTensor` | InvertibleModalityTransform | numpy -> torch 转换 |
| `StateActionTransform` | InvertibleModalityTransform | 核心变换：旋转转换 + 归一化 |
| `PerHorizonActionTransform` | InvertibleModalityTransform | 逐 horizon 归一化变换 |
| `StateActionPerturbation` | ModalityTransform | 高斯噪声扰动（仅训练） |
| `StateActionDropout` | ModalityTransform | 随机 Dropout（仅训练） |
| `StateActionSinCosTransform` | ModalityTransform | Sin-Cos 角度编码 |

## 与其他模块的关系

- **继承 `base.py`** - 继承 `ModalityTransform` 和 `InvertibleModalityTransform`
- **依赖 `schema`** - 使用 `DatasetMetadata`、`RotationType`、`StateActionMetadata`
- **与 `concat.py` 交互** - `ConcatTransform` 检查管线中的 `StateActionTransform` 以获取目标旋转信息
- **依赖 pytorch3d** - `RotationTransform` 使用 `pytorch3d.transforms` 进行旋转转换

## 总结

`state_action.py` 提供了状态和动作数据的全面预处理能力。其核心设计思路是将旋转转换和归一化解耦：先统一旋转表示，再进行标准化归一化。多种归一化模式（q99/mean_std/min_max/binary/scale）适应了不同类型数据的特性。PerHorizon 机制是 DreamZero 的特色功能，允许对不同预测步使用不同的归一化参数，这对于长 horizon 预测尤为重要。数据增强类（Perturbation/Dropout/SinCos）则为提升模型鲁棒性提供了工具。
