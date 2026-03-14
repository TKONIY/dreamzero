# `utils.py` -- 通用工具函数

## 文件概述

本文件提供了模型权重加载、状态字典处理和设备管理等通用工具函数，被 DreamZero 的多个模块共享使用。

## 关键代码解析

### init_weights_on_device -- Meta 设备初始化

```python
@contextmanager
def init_weights_on_device(device=torch.device("meta"), include_buffers=False):
    old_register_parameter = torch.nn.Module.register_parameter

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            module._parameters[name] = param_cls(
                module._parameters[name].to(device), **kwargs)

    torch.nn.Module.register_parameter = register_empty_parameter
    yield
    torch.nn.Module.register_parameter = old_register_parameter
```

上下文管理器，在其作用域内创建的所有模型参数都会被分配到指定设备（默认 `meta` 设备，即不分配实际内存）。这用于创建模型结构而不占用内存，之后再通过 `load_state_dict` 加载权重。

### 状态字典加载函数

```python
def load_state_dict(file_path, torch_dtype=None, device="cpu"):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype, device)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype, device)

def load_state_dict_from_folder(file_path, torch_dtype=None):
    for file_name in os.listdir(file_path):
        if file_name.endswith((".safetensors", ".bin", ".ckpt", ".pth", ".pt")):
            state_dict.update(load_state_dict(os.path.join(file_path, file_name)))
    return state_dict
```

### hash_state_dict_keys -- 模型配置自动识别

```python
def hash_state_dict_keys(state_dict, with_shape=True):
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape)
    return hashlib.md5(keys_str.encode("UTF-8")).hexdigest()
```

通过对状态字典的所有键名（和可选的形状信息）计算 MD5 哈希，自动识别模型配置。这使得加载权重时无需手动指定模型架构参数。

### 调试工具

```python
def search_parameter(param, state_dict):
    """在状态字典中搜索与给定参数匹配的键"""

def build_rename_dict(source_state_dict, target_state_dict, split_qkv=False):
    """构建两个状态字典之间的键名映射"""
```

## 核心类/函数表格

| 函数名 | 功能描述 |
|--------|---------|
| `init_weights_on_device()` | Meta 设备上下文管理器 |
| `load_state_dict()` | 加载 safetensors/bin 格式权重 |
| `load_state_dict_from_folder()` | 从文件夹加载所有权重 |
| `hash_state_dict_keys()` | 计算状态字典键的 MD5 哈希 |
| `search_parameter()` | 按参数值搜索键名 |
| `build_rename_dict()` | 构建键名映射字典 |
| `search_for_files()` | 递归搜索特定扩展名文件 |

## 与其他模块的关系

- **被 `vram_management.py` 使用**: `init_weights_on_device` 用于无内存开销地创建包装层
- **被 `wan_video_dit.py` 使用**: `hash_state_dict_keys` 用于自动识别模型配置
- **被 `wan_flow_matching_action_tf.py` 使用**: `load_state_dict` 用于加载预训练权重

## 总结

`utils.py` 是 DreamZero 的通用工具库，提供了模型权重管理的核心功能。`init_weights_on_device` 和 `hash_state_dict_keys` 是其最重要的两个工具，前者实现了零内存的模型创建，后者实现了自动化的模型配置识别。
