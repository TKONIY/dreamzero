# array_tensor_utils.py - 数组与张量通用操作工具

## 文件概述

`array_tensor_utils.py` 提供了一套同时兼容 NumPy 数组和 PyTorch 张量的通用操作函数。该模块的核心设计理念是**类型无关**（type-agnostic）——每个函数内部通过类型判断分别调用 NumPy 或 PyTorch 的对应 API，使调用者无需关心数据的底层类型。

大多数函数使用 `@make_recursive_func` 装饰器，能够自动处理任意嵌套的字典/列表/元组结构。

## 关键代码解析

### 1. 通用堆叠与拼接

```python
def any_stack(xs: List, *, dim: int = 0):
def any_concat(xs: List, *, dim: int = 0):
```

`any_stack` 对一组嵌套结构进行逐叶节点堆叠（类似 `np.stack` / `torch.stack`）。使用 `tree.map_structure` 在结构层面对齐，然后在叶节点层面执行堆叠。特别处理了 `float` 类型，默认转换为 `float32`。

`any_concat` 类似，但执行拼接（`np.concatenate` / `torch.cat`）。

### 2. 通用分块 `any_chunk`

```python
def any_chunk(x, chunks: int, *, dim: int = 0, strict: bool = True) -> List[Any]:
```

将嵌套结构中的所有数组/张量沿指定维度分块。实现较复杂：先创建 `chunks` 个结构副本，然后使用 `tree.map_structure_with_path` 遍历每个叶节点，执行分块后将结果分别赋值到对应的副本中。

### 3. 零值/一值/填充操作

```python
@make_recursive_func
def any_zeros_like(x):   # 创建同形零值对象
def any_ones_like(x):     # 创建同形一值对象
def any_zero_(x):         # 原地置零
def any_fill_(x, value):  # 原地填充
```

这些函数都使用 `@make_recursive_func`，支持对嵌套结构中的所有叶节点进行操作。特别地，`any_zeros_like` 和 `any_ones_like` 还支持 Python 原生数值类型（`int`、`float`、`np.number`），返回同类型的零/一值。

### 4. 批次大小获取

```python
def get_batch_size(x, strict: bool = False) -> int:
```

从任意嵌套结构中提取批次大小（第 0 维度的长度）。`strict=True` 时会检查所有叶节点的批次大小是否一致。

### 5. 描述与调试

```python
@make_recursive_func
def any_describe_str(x, shape_only=False):
```

生成数据结构的人类可读描述，对调试非常有用：
- NumPy 数组：`"np: [3, 224, 224] float32"`
- PyTorch 张量：`"torch: [32, 3, 224, 224] float32 cuda:0"`
- 标量：包含实际值
- 序列：`"list[5]"`

`any_describe(x, msg="")` 是其打印版本。

### 6. 切片与赋值

```python
@make_recursive_func
def any_slice(x, slice):        # 递归切片
def any_assign(x, assign_value, slice):  # 递归赋值
```

`any_slice` 对嵌套结构中的所有数组/张量应用相同的切片操作。`any_assign` 实现递归的 `x[slice] = value`，当两个结构不完全匹配时，以 `assign_value` 的结构为准。

### 7. 统计操作

```python
@make_recursive_func
def any_mean(x, dim=None, keepdim=False):
def any_variance(x, dim=None, keepdim=False, unbiased=False):
```

对嵌套结构中的所有数组/张量计算均值或方差。

## 核心类/函数表格

| 函数 | 说明 |
|------|------|
| `is_array_tensor(obj)` | 判断是否为 NumPy 数组或 PyTorch 张量 |
| `is_numpy(obj)` / `is_tensor(obj)` | 分别判断类型 |
| `any_stack(xs, dim)` | 嵌套结构逐叶节点堆叠 |
| `any_concat(xs, dim)` | 嵌套结构逐叶节点拼接 |
| `any_chunk(x, chunks, dim)` | 嵌套结构逐叶节点分块 |
| `chunk_seq(arr, chunks)` | 将序列分成 N 等份 |
| `any_zeros_like(x)` | 创建同形零值 |
| `any_ones_like(x)` | 创建同形一值 |
| `any_zero_(x)` | 原地置零 |
| `any_fill_(x, value)` | 原地填充 |
| `get_batch_size(x)` | 获取批次大小 |
| `add_batch_dim(x)` / `remove_batch_dim(x)` | 添加/移除批次维度 |
| `any_to_primitive(x)` | 转换为 Python 原生类型 |
| `any_get_shape(x)` | 获取形状 |
| `any_mean(x)` / `any_variance(x)` | 计算均值/方差 |
| `any_describe_str(x)` | 生成描述字符串 |
| `any_describe(x, msg)` | 打印描述信息 |
| `any_slice(x, slice)` | 递归切片 |
| `any_assign(x, assign_value, slice)` | 递归赋值 |
| `any_transpose_first_two_axes(x)` | 交换前两个轴 |

## 与其他模块的关系

- **依赖** `tree_utils.py`：使用 `copy_non_leaf`、`is_sequence`、`tree_assign_at_path`、`tree_value_at_path`
- **依赖** `functional_utils.py`：使用 `make_recursive_func` 实现递归操作
- **被** `torch_utils.py` 依赖：`RunningMeanStd` 中使用 `any_mean`、`any_variance`、`get_batch_size`
- **被** `image_utils.py` 依赖：使用 `any_describe` 进行调试
- **被数据处理流程广泛使用**：在模型训练的数据加载、预处理、后处理阶段频繁使用

## 总结

`array_tensor_utils.py` 通过类型判断分发机制，实现了 NumPy 和 PyTorch 操作的统一抽象。配合 `@make_recursive_func` 装饰器，这些函数能够透明地处理任意深度嵌套的数据结构。这种设计在机器人学习项目中特别有价值——观测数据通常是复杂的嵌套字典（包含图像、关节角度、力传感器等），而本模块让开发者可以用一行代码对整个数据结构进行堆叠、切片、转移设备等操作。
