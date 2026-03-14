# 拼接变换 (`groot/vla/data/transform/concat.py`)

## 文件概述

`concat.py` 实现了 `ConcatTransform` 类，负责将多个分散的模态键（如 `video.ego_view`、`state.joint_position`、`action.joint_position` 等）按指定顺序拼接为统一的张量（`video`、`state`、`action`）。这是数据预处理管线中最后执行的变换之一，将字典格式的数据转换为模型可直接消费的格式。

## 关键代码解析

### `ConcatTransform` 类

```python
class ConcatTransform(InvertibleModalityTransform):
    video_concat_order: list[str]              # 视频拼接顺序
    state_concat_order: Optional[list[str]]    # 状态拼接顺序
    action_concat_order: Optional[list[str]]   # 动作拼接顺序
    action_dims: dict[str, int]                # 各动作键的维度
    state_dims: dict[str, int]                 # 各状态键的维度
```

#### `apply` 方法（正向拼接）

分三步处理视频、状态和动作：

**视频拼接：**
```python
# 将各视图沿新维度拼接: [..., H, W, C] -> [..., V, H, W, C]
for video_key in self.video_concat_order:
    video_data = data.pop(video_key)
    unsqueezed_video = np.expand_dims(video_data, axis=-4)
unsqueezed_video = np.concatenate(unsqueezed_videos, axis=-4)
data["video"] = unsqueezed_video
```

**状态拼接：**
```python
# 使用 torch.cat 沿最后一维拼接: [T, D1], [T, D2], ... -> [T, D_state]
data["state"] = torch.cat([data.pop(key) for key in self.state_concat_order], dim=-1)
```

**动作拼接：**
```python
# 同状态拼接: [T, D1], [T, D2], ... -> [T, D_action]
data["action"] = torch.cat([data.pop(key) for key in self.action_concat_order], dim=-1)
```

#### `unapply` 方法（逆向拆分）

推理时将模型输出的拼接动作张量拆分回各分量：

```python
def unapply(self, data):
    action_tensor = data.pop("action")
    for key in self.action_concat_order:
        end_dim = start_dim + self.get_state_action_dims_post_transform(key)
        data[key] = action_tensor[..., start_dim:end_dim]
        start_dim = end_dim
```

#### 维度感知

`ConcatTransform` 需要知道变换后的正确维度，因为旋转转换可能改变维度（例如四元数 4D 转轴角 3D）：

```python
def get_state_action_dims_post_transform(self, key):
    if self.is_rotation_key(key):
        target_rotations = self._get_target_rotations_from_pipeline()
        if key in target_rotations:
            if target_rotation == "axis_angle": return 3
            elif target_rotation == "quaternion": return 4
            elif target_rotation == "rotation_6d": return 6
```

这就是为什么 `ComposedModalityTransform.set_metadata()` 会调用 `set_transform_pipeline()` -- `ConcatTransform` 需要检查管线中其他变换是否修改了维度。

#### `set_metadata` 方法

在设置元数据时预计算各键的维度：

```python
def set_metadata(self, dataset_metadata):
    super().set_metadata(dataset_metadata)
    for key in self.action_concat_order:
        self.action_dims[key] = self.get_state_action_dims(key)
    for key in self.state_concat_order:
        self.state_dims[key] = self.get_state_action_dims(key)
```

### 维度校验

`apply` 方法中包含严格的维度校验，确保输入数据的维度符合预期：

```python
target_shapes = [self.action_dims[key]]
if self.is_rotation_key(key):
    target_shapes.extend([3, 4, 6])  # 支持各种旋转表示
assert data[key].shape[-1] in target_shapes
```

## 核心类/函数表格

| 名称 | 类型 | 说明 |
|------|------|------|
| `ConcatTransform` | InvertibleModalityTransform | 将分散的模态键拼接为统一张量 |
| `apply` | 方法 | 正向拼接：video/state/action 键合并 |
| `unapply` | 方法 | 逆向拆分：action/state 张量按键拆分 |
| `set_transform_pipeline` | 方法 | 接收完整变换管线用于维度推断 |
| `get_state_action_dims_post_transform` | 方法 | 获取变换后的维度（考虑旋转转换） |

## 与其他模块的关系

- **继承 `base.py`** - 继承 `InvertibleModalityTransform`
- **依赖 `schema`** - 使用 `DatasetMetadata` 和 `StateActionMetadata` 获取维度信息
- **与 `state_action.py` 交互** - 检查管线中的旋转变换以确定变换后维度

## 总结

`ConcatTransform` 是变换管线中的"汇聚点"，将多个分散的模态数据合并为模型所需的统一格式。其最精妙的设计在于维度感知机制：通过检查整个变换管线来推断旋转转换后的正确维度，确保拼接和拆分操作的正确性。这使得系统可以灵活地在不同旋转表示之间切换，而无需手动调整拼接配置。
