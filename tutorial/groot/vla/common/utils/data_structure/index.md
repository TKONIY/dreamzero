# groot.vla.common.utils.data_structure 模块

## 模块概述

`data_structure` 子包提供嵌套数据结构操作和神经网络形状推断工具。通过 `__init__.py` 导出两个子模块：

```python
from .shape_utils import *
from .tree_utils import *
```

## 文件列表

| 文件 | 说明 | 教程链接 |
|------|------|----------|
| `tree_utils.py` | 嵌套数据结构的遍历、映射、堆叠、拆分和广播 | [详细教程](tree_utils.md) |
| `shape_utils.py` | 卷积、池化、切片操作的输出形状推断 | [详细教程](shape_utils.md) |

## 模块间关系

`tree_utils.py` 是整个 `common/utils` 工具库中被依赖最广泛的文件：

```
tree_utils.py
├── functional_utils.py [misc] (is_mapping, is_sequence)
├── array_tensor_utils.py [misc] (copy_non_leaf, tree_assign_at_path, tree_value_at_path)
├── file_utils.py [io] (is_sequence)
├── torch_utils.py [misc] (tree_value_at_path)
└── 间接被几乎所有其他模块依赖
```

`shape_utils.py` 相对独立，不依赖项目内其他模块。

## 核心设计理念

### tree_utils.py -- 嵌套结构即一等公民

在机器人学习中，数据天然具有嵌套结构。例如一个观测可能是：

```python
obs = {
    "image": {"front": np.array(...), "wrist": np.array(...)},
    "proprioception": {"joint_pos": np.array(...), "gripper": np.array(...)},
}
```

`tree_utils.py` 让你可以像操作单个数组一样操作整个嵌套结构：

```python
# 将一批观测堆叠为批次
batch_obs = stack_sequence_fields([obs1, obs2, obs3])

# 按路径访问特定值
front_cam = tree_value_at_path(batch_obs, ("image", "front"))
```

### shape_utils.py -- 无实例化的形状计算

通过数学公式而非实际张量运算推断形状，这在以下场景中非常有用：

- 构建编码器后自动确定全连接层输入维度
- 验证网络结构的维度一致性
- 在配置中声明性地定义网络架构
