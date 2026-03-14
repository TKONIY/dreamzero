# shape_utils.py - 形状推断工具

## 文件概述

`shape_utils.py` 提供了卷积层、池化层和切片操作的输出形状推断函数。这些工具允许在不实际创建张量的情况下计算神经网络各层的输出形状，对于构建动态网络结构（如自动计算全连接层的输入维度）非常有用。

## 关键代码解析

### 1. 通用卷积形状推断

```python
def shape_convnd(dim, input_shape, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, has_batch=False):
```

根据 PyTorch 的卷积公式计算输出形状：

```
output_size[i] = floor((input_size[i] + 2*padding[i] - dilation[i]*(kernel_size[i]-1) - 1) / stride[i] + 1)
```

支持 1D 到 3D 卷积，`has_batch=True` 时输入形状包含 batch 维度。内部使用 `_expands` 函数将标量参数（如 `kernel_size=3`）扩展为对应维度的元组。

通过 `functools.partial` 生成特化版本：
```python
shape_conv1d = partial(shape_convnd, 1)
shape_conv2d = partial(shape_convnd, 2)
shape_conv3d = partial(shape_convnd, 3)
```

### 2. 转置卷积形状推断

```python
def shape_transpose_convnd(dim, input_shape, out_channels, kernel_size, stride=1,
                           padding=0, output_padding=0, dilation=1, has_batch=False):
```

使用转置卷积的形状公式：

```
output_size[i] = (input_size[i] - 1) * stride[i] - 2*padding[i] + kernel_size[i] + output_padding[i]
```

### 3. 池化层形状推断

```python
def shape_poolnd(dim, input_shape, kernel_size, stride=None, padding=0, dilation=1, has_batch=False):
```

与卷积的区别：输出通道数不变（保持输入通道数），`stride` 默认为 `kernel_size`（PyTorch 池化层的约定）。

同样通过 partial 生成 `shape_maxpool1d/2d/3d` 和 `shape_avgpool1d/2d/3d`（avgpool 固定 `dilation=1`）。

### 4. 切片形状推断

```python
_HELPER_TENSOR = torch.zeros((1,))

def shape_slice(input_shape, slice):
```

这是一个巧妙的技巧（归功于 Adam Paszke）：使用 `torch.Tensor.expand()` 创建一个虚拟张量（不实际分配内存），然后对其应用切片操作，通过检查结果的形状得到切片后的输出形状。

```python
shape = _HELPER_TENSOR.expand(*input_shape)[slice]
```

`ShapeSlice` 类封装了这个操作，提供更自然的 `[]` 语法：
```python
s = ShapeSlice((3, 224, 224))
s[:, ::2, ::2]  # -> (3, 112, 112)
```

### 5. 形状检查

```python
def check_shape(value, expected, err_msg="", mode="raise"):
```

验证张量/数组的形状是否符合预期。`expected` 中的 `None` 值表示该维度可以匹配任意大小。支持三种模式：`"raise"`（抛异常）、`"return"`（返回布尔值）、`"warning"`（发出警告）。

## 核心类/函数表格

| 函数/类 | 说明 |
|---------|------|
| `shape_convnd(dim, ...)` | 通用 N-D 卷积输出形状推断 |
| `shape_conv1d/2d/3d(...)` | 1D/2D/3D 卷积形状推断 |
| `shape_transpose_convnd(dim, ...)` | 通用 N-D 转置卷积形状推断 |
| `shape_transpose_conv1d/2d/3d(...)` | 1D/2D/3D 转置卷积形状推断 |
| `shape_poolnd(dim, ...)` | 通用 N-D 池化输出形状推断 |
| `shape_maxpool1d/2d/3d(...)` | 最大池化形状推断 |
| `shape_avgpool1d/2d/3d(...)` | 平均池化形状推断 |
| `shape_slice(input_shape, slice)` | 切片操作形状推断 |
| `ShapeSlice` 类 | 支持 `[]` 语法的切片形状推断 |
| `check_shape(value, expected)` | 验证张量形状是否符合预期 |

## 与其他模块的关系

- **外部依赖**：`numpy`、`torch`
- **独立性强**：不依赖项目内的其他模块
- **被网络定义代码使用**：在构建神经网络时自动计算中间层的形状
- **通过** `data_structure/__init__.py` 导出

## 总结

`shape_utils.py` 提供了一套完整的神经网络层形状推断工具。通过公式计算而非实际前向传播来确定输出形状，这在构建复杂的动态网络结构时非常有价值——例如在卷积编码器之后需要知道特征图大小来设置全连接层的输入维度。`shape_slice` 的零内存开销技巧和 `check_shape` 的灵活断言模式体现了实用性优先的设计理念。
