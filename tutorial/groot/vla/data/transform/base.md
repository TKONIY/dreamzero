# 变换基类定义 (`groot/vla/data/transform/base.py`)

## 文件概述

`base.py` 定义了 DreamZero 数据变换管线的基础架构，包含抽象基类 `ModalityTransform`、可逆变换接口 `InvertibleModalityTransform`、恒等变换 `IdentityModalityTransform` 以及核心的组合变换 `ComposedModalityTransform`。这些类共同构成了一个灵活、可扩展的数据预处理框架。

## 关键代码解析

### 1. `ModalityTransform` 抽象基类

```python
class ModalityTransform(BaseModel, ABC):
    apply_to: list[str]     # 需要应用变换的数据键列表
    training: bool = True   # 训练/评估模式标志
    _dataset_metadata: DatasetMetadata | None = PrivateAttr(default=None)
```

核心设计要点：

- **双重继承** - 同时继承 `BaseModel`（Pydantic）和 `ABC`（抽象类），兼具配置序列化和接口约束能力
- **`apply_to` 字段** - 指定变换作用的数据键（如 `["video.ego_view", "video.side_view"]`）
- **`_dataset_metadata`** - 通过 `PrivateAttr` 存储数据集元数据，不参与序列化
- **`__call__`** - 委托给 `apply()` 方法，使变换实例可以作为函数调用
- **`train()`/`eval()`** - 切换训练/评估模式，某些变换仅在训练时生效（如数据增强）

`set_metadata()` 方法允许在创建变换后动态注入数据集元数据：

```python
def set_metadata(self, dataset_metadata: DatasetMetadata):
    self.dataset_metadata = dataset_metadata
```

### 2. `InvertibleModalityTransform`

```python
class InvertibleModalityTransform(ModalityTransform):
    @abstractmethod
    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        pass
```

在 `ModalityTransform` 基础上增加了 `unapply()` 方法，支持逆向变换。这对于推理时将模型预测的归一化动作还原为原始值至关重要。

### 3. `IdentityModalityTransform`

```python
class IdentityModalityTransform(ModalityTransform):
    def apply(self, data):
        return data
    def unapply(self, data):
        return data
```

恒等变换，不修改数据。可用作占位符或调试。

### 4. `ComposedModalityTransform`（核心组合器）

```python
class ComposedModalityTransform(ModalityTransform):
    transforms: list[ModalityTransform]
```

将多个变换组合为一个管线，按顺序执行。这是 DreamZero 中最常用的变换容器。

#### `set_metadata` 方法

```python
def set_metadata(self, dataset_metadata: DatasetMetadata):
    for transform in self.transforms:
        transform.set_metadata(dataset_metadata)
        if hasattr(transform, "set_transform_pipeline"):
            getattr(transform, "set_transform_pipeline")(self.transforms)
```

关键细节：除了传递元数据外，还会调用 `set_transform_pipeline`（如果变换支持）。这是为 `ConcatTransform` 设计的机制 -- 拼接变换需要知道管线中是否存在旋转转换等改变维度的变换，以正确计算拼接后的维度。

#### `apply` 方法

按顺序执行所有变换，并提供错误信息追踪：

```python
def apply(self, data):
    for i, transform in enumerate(self.transforms):
        try:
            data = transform(data)
        except Exception as e:
            raise ValueError(f"Error applying transform {i} to data: {e}") from e
    return data
```

#### `unapply` 方法

**逆序**执行可逆变换的 `unapply`：

```python
def unapply(self, data):
    for i, transform in enumerate(reversed(self.transforms)):
        if isinstance(transform, InvertibleModalityTransform):
            data = transform.unapply(data)
    return data
```

注意只有 `InvertibleModalityTransform` 的实例才会被逆向执行，非可逆变换（如数据增强）会被自动跳过。

#### `set_per_horizon_statistics` 方法

```python
def set_per_horizon_statistics(self, per_horizon_stats):
    for transform in self.transforms:
        if hasattr(transform, "set_per_horizon_statistics"):
            transform.set_per_horizon_statistics(per_horizon_stats)
```

为支持 per-horizon 归一化的变换传递逐 horizon 步骤的统计数据。

## 核心类/函数表格

| 名称 | 类型 | 说明 |
|------|------|------|
| `ModalityTransform` | 抽象基类 | 所有变换的基类，定义 `apply()` 接口 |
| `InvertibleModalityTransform` | 抽象基类 | 可逆变换基类，增加 `unapply()` 接口 |
| `IdentityModalityTransform` | 具体类 | 恒等变换，不修改数据 |
| `ComposedModalityTransform` | 具体类 | 组合变换，按序执行多个变换，支持正逆向 |

## 与其他模块的关系

- **依赖 `schema`** - 使用 `DatasetMetadata` 存储数据集元数据
- **被 `concat.py`、`language.py`、`state_action.py`、`video.py` 继承** - 作为所有具体变换的基类
- **被 `dataset/lerobot.py` 使用** - `LeRobotSingleDataset` 持有 `ComposedModalityTransform` 实例

## 总结

`base.py` 建立了 DreamZero 变换管线的核心抽象。`ModalityTransform` 通过 `apply_to` 字段实现了"声明式"变换目标选择，`ComposedModalityTransform` 通过管线组合实现了灵活的变换编排。正向/逆向变换的对称设计使得训练时的数据预处理和推理时的动作还原共享同一套变换定义，避免了代码重复和不一致性。
