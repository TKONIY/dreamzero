# groot.vla.common.utils.io 模块

## 模块概述

`io` 子包负责所有输入/输出相关的工具功能，包括文件系统操作、数据格式读写、配置管理以及终端输出控制。通过 `__init__.py` 统一导出所有子模块的公开符号：

```python
from .config_utils import *
from .file_utils import *
from .hdf5_utils import *
from .json_utils import *
from .print_utils import *
from .termcolor import *
```

## 文件列表

| 文件 | 说明 | 教程链接 |
|------|------|----------|
| `config_utils.py` | Hydra/OmegaConf 配置管理，类注册与实例化系统 | [详细教程](config_utils.md) |
| `file_utils.py` | 文件系统操作工具集（路径、目录、文件读写、归档） | [详细教程](file_utils.md) |
| `hdf5_utils.py` | HDF5 格式的递归保存、加载和比较 | [详细教程](hdf5_utils.md) |
| `json_utils.py` | JSON、YAML、JSONL 格式的读写工具和 Jsonl 流式读写器 | [详细教程](json_utils.md) |
| `print_utils.py` | 打印格式化、调试打印器、输出重定向、日志过滤器 | [详细教程](print_utils.md) |
| `termcolor.py` | 终端 ANSI 彩色文本输出 | [详细教程](termcolor.md) |

## 模块间关系

```
termcolor.py (独立)
hdf5_utils.py (独立，仅依赖外部库)

file_utils.py ← json_utils.py (路径处理)
file_utils.py ← config_utils.py (路径处理)

misc/functional_utils.py ← config_utils.py (装饰器)
misc/functional_utils.py ← json_utils.py (make_recursive_func)
misc/functional_utils.py ← print_utils.py (meta_decorator)
misc/misc_utils.py ← print_utils.py (match_patterns)
```

## 核心使用场景

1. **配置加载与对象构造**：`config_utils.py` 的 `instantiate()` 从 YAML 配置构建模型
2. **数据持久化**：`json_utils.py` 保存训练日志，`hdf5_utils.py` 保存演示数据
3. **文件管理**：`file_utils.py` 的 `f_join`、`f_mkdir` 管理实验目录
4. **调试输出**：`print_utils.py` 的 `DebugPrinter` 打印张量形状摘要
