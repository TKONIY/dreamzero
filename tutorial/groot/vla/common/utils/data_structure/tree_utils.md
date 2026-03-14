# tree_utils.py - 嵌套数据结构工具

## 文件概述

`tree_utils.py` 提供了处理嵌套数据结构（如嵌套字典、列表、元组）的工具函数。该模块建立在 DeepMind 的 [dm-tree](https://tree.readthedocs.io/) 库之上，扩展了其功能，并包含了源自 [Acme](https://github.com/deepmind/acme) 项目的工具函数（Apache 2.0 许可证）。

在深度学习和机器人学习中，数据通常以嵌套字典的形式组织（如 `{"obs": {"image": ..., "state": ...}, "action": ...}`），本模块提供了对这种结构进行批量操作的核心能力。

## 关键代码解析

### 1. 基础类型判断

```python
def is_sequence(obj):
    return isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str)

def is_mapping(obj):
    return isinstance(obj, collections.abc.Mapping)
```

这两个函数是整个工具库中被引用最多的基础函数。`is_sequence` 特别排除了字符串（字符串虽然是 `Sequence`，但通常不应被视为可遍历的嵌套结构）。

### 2. 路径访问与赋值

```python
def tree_value_at_path(obj, paths: Tuple):
def tree_assign_at_path(obj, paths: Tuple, value):
```

`tree_value_at_path` 按路径元组逐层深入嵌套结构取值。路径格式由 `dm-tree` 的 `map_structure_with_path` 回调提供，通常是 `(key1, key2, ...)` 的元组。

`tree_assign_at_path` 是其赋值版本，遍历到倒数第二层后在最后一个键处赋值。

### 3. 结构复制

```python
def copy_non_leaf(obj):
    return tree.map_structure(lambda x: x, obj)
```

一个精巧的实现——通过 `map_structure` 和恒等函数，深拷贝了嵌套结构的容器层（字典、列表等），但叶节点（如张量）仍然是引用。这在需要修改结构但不想复制大型张量时非常有用。

### 4. 高性能结构映射

```python
def fast_map_structure(func, *structure):
    flat_structure = (tree.flatten(s) for s in structure)
    entries = zip(*flat_structure)
    return tree.unflatten_as(structure[-1], [func(*x) for x in entries])
```

跳过了 `tree.map_structure` 中的部分错误检查，用于对性能敏感的场景。

### 5. 堆叠与拆分（源自 Acme）

```python
def stack_sequence_fields(sequence):
```

将一组结构相同的嵌套对象堆叠为一个嵌套对象，叶节点变成堆叠后的数组。例如：

```python
输入: [{"action": [1.0], "reward": 1.0}, {"action": [0.5], "reward": 0.0}]
输出: {"action": array([[1.0], [0.5]]), "reward": array([1.0, 0.0])}
```

如果叶节点形状不一致（无法 `np.stack`），回退到 `np.asarray`。

```python
def unstack_sequence_fields(struct, batch_size):
```

反向操作，将批次化的嵌套结构拆分为单个样本的列表。

### 6. 结构广播

```python
def broadcast_structures(*args):
```

将非嵌套值广播为与嵌套值相同的结构。例如：

```python
a = ('a', 'b')
b = 'c'
tree_a, tree_b = broadcast_structures(a, b)
# tree_b = ('c', 'c')
```

这对于将标量参数应用到嵌套结构的每个叶节点非常有用。

## 核心类/函数表格

| 函数 | 说明 |
|------|------|
| `is_sequence(obj)` | 判断是否为非字符串序列 |
| `is_mapping(obj)` | 判断是否为映射类型 |
| `tree_value_at_path(obj, paths)` | 按路径元组访问嵌套结构中的值 |
| `tree_assign_at_path(obj, paths, value)` | 按路径元组在嵌套结构中赋值 |
| `copy_non_leaf(obj)` | 深拷贝结构容器但不拷贝叶节点 |
| `fast_map_structure(func, *structure)` | 高性能版 map_structure |
| `stack_sequence_fields(sequence)` | 将嵌套对象列表堆叠为批次化结构 |
| `unstack_sequence_fields(struct, batch_size)` | 将批次化结构拆分为对象列表 |
| `broadcast_structures(*args)` | 将非嵌套值广播为嵌套结构 |

## 与其他模块的关系

- **外部依赖**：`dm-tree`（核心树操作库）、`numpy`
- **被几乎所有工具模块依赖**：
  - `functional_utils.py` 导入 `is_mapping`、`is_sequence`
  - `array_tensor_utils.py` 导入 `copy_non_leaf`、`tree_assign_at_path`、`tree_value_at_path`
  - `file_utils.py` 导入 `is_sequence`
  - `config_utils.py` 间接使用（通过 `functional_utils.py`）
  - `torch_utils.py` 导入 `tree_value_at_path`
- **作为数据结构处理的基础层**，在数据加载、模型前向传播、经验回放等环节被广泛使用

## 总结

`tree_utils.py` 是整个工具库的数据结构处理基石。`is_sequence` 和 `is_mapping` 是被引用最广泛的基础判断函数；`stack_sequence_fields` 和 `unstack_sequence_fields` 在数据加载（将多个样本组成批次）和经验回放（将批次拆分为单个时间步）中不可或缺；`tree_value_at_path` 和 `tree_assign_at_path` 为嵌套结构的精确操作提供了路径化访问能力。整个模块基于 `dm-tree` 构建，继承了其在处理任意嵌套数据结构方面的强大能力。
