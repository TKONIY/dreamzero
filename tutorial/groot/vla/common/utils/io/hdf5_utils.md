# hdf5_utils.py - HDF5 数据工具

## 文件概述

`hdf5_utils.py` 提供了基于 [h5py](https://www.h5py.org/) 的 HDF5 文件递归读写与比较工具。HDF5 是一种专为大规模科学数据设计的层次化文件格式，特别适合存储机器人操作中的多模态演示数据（如图像、动作序列、传感器读数等）。

该模块的核心设计思想是将嵌套的 Python 字典结构（或 Pydantic 模型）与 HDF5 的层次化 Group/Dataset 结构建立双向映射。

## 关键代码解析

### 1. 递归保存 `hdf5_save`

```python
def hdf5_save(data: BaseModel | dict, group: h5py.Group) -> None:
```

递归遍历 Pydantic 模型或字典：
- **`np.ndarray`** 直接创建为 HDF5 Dataset
- **嵌套的字典/模型** 创建为 HDF5 Group，然后递归处理
- **基本类型**（int、float、list 等）先转换为 `np.array` 再存储

如果传入的是 Pydantic `BaseModel`，会先通过 `model_dump(mode="python", exclude_none=True)` 转为字典，排除所有 `None` 值字段。

### 2. 递归加载 `hdf5_load`

```python
def hdf5_load(group: h5py.Group) -> dict:
```

遍历 HDF5 Group 的所有条目：
- `h5py.Dataset` 通过 `value[()]` 读取为 NumPy 数组
- `h5py.Group` 递归加载为嵌套字典

加载结果始终是 Python 原生字典，不会自动还原为 Pydantic 模型。

### 3. 子集判断与等价判断

```python
def hdf5_is_subset(this: h5py.Group, other: h5py.Group, verbose=False) -> bool:
def hdf5_is_equal(this: h5py.Group, other: h5py.Group, verbose=False) -> bool:
```

`hdf5_is_subset` 递归检查 `this` 中的每个键/值是否都存在于 `other` 中且值相等。它区分三种 HDF5 节点类型：
- **Group**：递归检查子集关系
- **Dataset**：使用 `np.array_equal` 逐元素比较
- **Datatype**：直接比较

`hdf5_is_equal` 通过双向子集检查实现等价判断：`A == B` 等价于 `A ⊆ B ∧ B ⊆ A`。

`verbose=True` 时会打印不匹配的具体键名，便于调试。

## 核心类/函数表格

| 函数 | 说明 |
|------|------|
| `hdf5_save(data, group)` | 将字典或 Pydantic 模型递归保存到 HDF5 Group |
| `hdf5_load(group)` | 从 HDF5 Group 递归加载为 Python 字典 |
| `hdf5_is_subset(this, other)` | 检查一个 HDF5 Group 是否是另一个的子集 |
| `hdf5_is_equal(this, other)` | 检查两个 HDF5 Group 是否内容完全一致 |

## 与其他模块的关系

- **外部依赖**：`h5py`（HDF5 读写）、`numpy`（数组操作）、`pydantic`（数据模型）
- **在项目中的位置**：主要服务于数据集的存储与加载流程，特别是机器人演示数据的序列化
- **与 `file_utils.py` 互补**：`file_utils.py` 处理通用文件操作，`hdf5_utils.py` 专门处理 HDF5 格式

## 总结

`hdf5_utils.py` 提供了简洁的 HDF5 递归序列化接口，将复杂的嵌套数据结构与 HDF5 的层次化存储格式无缝对接。通过支持 Pydantic 模型，它可以与项目中的数据模型定义紧密配合。子集和等价判断功能对于数据验证和测试非常有用，确保数据在序列化/反序列化过程中保持一致性。
