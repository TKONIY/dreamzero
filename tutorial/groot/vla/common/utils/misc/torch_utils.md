# torch_utils.py - PyTorch 工具集

## 文件概述

`torch_utils.py` 是整个工具库中最大的文件，提供了全面的 PyTorch 辅助工具。功能涵盖：

1. **随机种子与确定性**：全局种子设置、确定性训练配置
2. **模型参数管理**：冻结/解冻、梯度裁剪、参数计数
3. **状态字典操作**：保存/加载/转移状态字典
4. **模型加载与保存**：PyTorch checkpoint 的读写
5. **张量操作**：多维索引、归一化、one-hot 编码
6. **分布式训练**：DDP 模型解包、方法包装
7. **统计工具**：运行均值/标准差、平均值计量器
8. **分类评估**：top-k 准确率计算

## 关键代码解析

### 1. 随机种子管理

```python
def set_seed_everywhere(seed, deterministic=False, set_tensorflow=False,
                        handle_invalid_seed="none") -> Optional[int]:
```

一站式设置所有随机种子源：Python `random`、`numpy`、PyTorch（CPU 和 GPU）、`PYTHONHASHSEED` 环境变量，以及可选的 TensorFlow。`get_seed` 函数处理种子的各种输入形式（整数、`"system"`、`None`、负数）。

`set_deterministic` 配置 CUDA 和 cuDNN 的确定性模式，包括设置 `CUBLAS_WORKSPACE_CONFIG` 环境变量。

### 2. 模型参数控制

```python
def freeze_params(model):      # 冻结参数 + 设置 eval 模式
def unfreeze_params(model):    # 解冻参数 + 设置 train 模式
def clip_grad_norm(model, max_norm, norm_type=2):  # 梯度范数裁剪
def clip_grad_value(model, max_value):             # 梯度值裁剪
```

`freeze_params` 同时设置 `requires_grad=False` 和 `model.eval()`，确保 BatchNorm 等层也切换到推理模式。

### 3. 状态字典操作

```python
def to_state_dict(objects, to_cpu=False, copy=False, unwrap_ddp=False):
```

递归遍历嵌套结构，对所有实现了 `state_dict()` 方法的对象提取状态字典。支持：
- `to_cpu=True`：将所有张量复制到 CPU（用于保存 checkpoint）
- `unwrap_ddp=True`：自动解包 DDP 模型
- 对 GPU 张量执行 `detach()` 避免梯度图泄漏

```python
def load_state_dict(objects, states, strip_prefix=None, strict=False):
```

反向操作，将状态字典加载回对象。`strip_prefix` 参数支持去除键前缀（如从 `"module."` 前缀的 DDP checkpoint 加载到普通模型）。

### 4. 分布式训练支持

```python
class DDPMethodWrapper(nn.Module):
    def __init__(self, net: nn.Module, method_name: str):
```

DDP 只会并行化 `forward()` 方法。`DDPMethodWrapper` 将其他方法（如 `compute_loss`）包装为 `forward()`，从而可以被 DDP 并行化。它不保存额外参数（`state_dict()` 返回空字典）。

### 5. 图像归一化

```python
def torch_normalize(tensor, mean, std, inplace=False):
```

PyTorch 版的图像归一化，改编自 torchvision。关键改进：
- 支持任意维度的 mean/std（自动广播）
- 检查 std 中是否有零值（避免除零错误）

### 6. 运行统计量

```python
class RunningMeanStd:
    def update(self, values):
```

基于 Welford 并行算法的在线均值/方差计算器，支持 NumPy 和 PyTorch。适用于需要在线计算观测归一化统计量的强化学习场景。

```python
class AverageMeter:
    def update(self, value, n=1):
```

简单的均值计量器，常用于训练过程中跟踪损失值。

### 7. Top-k 分类准确率

```python
def classify_accuracy(output, target, topk=1, mask=None, reduction="mean"):
```

支持 top-k 准确率计算，可选的样本掩码（用于不等长序列），以及多种归约方式（mean/sum/none）。

### 8. 数据集分割

```python
def sequential_split_dataset(dataset, split_portions):
```

按比例顺序分割数据集（非随机），返回 `torch.utils.data.Subset` 列表。确保最后一份补齐因取整导致的差额。

## 核心类/函数表格

| 函数/类 | 说明 |
|---------|------|
| `weight_init(m)` | 正交初始化 Linear/Conv 层 |
| `get_seed(seed)` | 处理各种种子输入格式 |
| `set_deterministic(flag)` | 配置确定性训练环境 |
| `set_seed_everywhere(seed)` | 一站式设置所有随机种子 |
| `eval_mode(*models)` | 上下文管理器，临时切换 eval 模式 |
| `get_device(x)` | 获取嵌套结构的设备信息 |
| `load_torch` / `save_torch` | PyTorch checkpoint 读写（默认 CPU） |
| `torch_compute_stats(x)` | 计算张量统计摘要字符串 |
| `tensor_hash(x)` | 计算张量哈希值（用于调试） |
| `torch_flatten_indices` | 多维索引展平为一维 |
| `torch_multi_index_select` | 多维高级索引选择 |
| `freeze_params` / `unfreeze_params` | 冻结/解冻模型参数 |
| `clip_grad_norm` / `clip_grad_value` | 梯度裁剪 |
| `set_requires_grad(model, flag)` | 设置参数梯度需求 |
| `to_state_dict(objects)` | 递归提取状态字典 |
| `load_state_dict(objects, states)` | 递归加载状态字典 |
| `count_parameters(model)` | 统计模型参数数量 |
| `readable_count_parameters(model)` | 可读格式的参数数量 |
| `get_module_device(model)` | 获取模型所在设备 |
| `maybe_transfer_module(model, device)` | 按需转移模型到指定设备 |
| `clone_model(model)` | 深拷贝模型 |
| `update_soft_params(net, target, tau)` | 软更新目标网络参数 |
| `torch_normalize(tensor, mean, std)` | 图像归一化 |
| `contains_rnn(net)` | 检查模型是否包含 RNN |
| `multi_one_hot(x, num_classes)` | MultiDiscrete 空间的 one-hot 编码 |
| `random_derangement(n)` | 生成随机错排（对比学习用） |
| `classify_accuracy(output, target)` | Top-k 分类准确率 |
| `sequential_split_dataset(dataset, portions)` | 顺序分割数据集 |
| `DDPMethodWrapper` 类 | 将模型方法包装为 DDP 可用的 forward |
| `RunningMeanStd` 类 | 在线均值/标准差计算 |
| `AverageMeter` 类 | 均值计量器 |

## 与其他模块的关系

- **依赖** `tree_utils.py`：`tree_value_at_path` 用于状态字典路径访问
- **依赖** `file_utils.py`：`f_join` 用于路径处理
- **依赖** `print_utils.py`：`to_readable_count_str` 用于参数量格式化
- **依赖** `functional_utils.py`：`implements_method`、`accumulate` 等
- **被** `image_utils.py` 依赖：`torch_normalize` 用于图像预处理
- **被训练和评估代码广泛使用**

## 总结

`torch_utils.py` 是 DreamZero 项目中 PyTorch 相关操作的百宝箱。它解决了深度学习训练中的大量常见需求：从种子设置到模型管理，从状态字典操作到分布式训练支持。特别值得注意的是 `to_state_dict` / `load_state_dict` 的递归设计——它们可以一次性处理包含多个模型、优化器和调度器的复杂嵌套结构，极大简化了 checkpoint 管理代码。`RunningMeanStd` 和 `update_soft_params` 则体现了该文件对强化学习场景的专门支持。
