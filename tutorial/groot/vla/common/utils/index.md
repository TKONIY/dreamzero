# groot.vla.common.utils 模块

## 模块概述

`utils` 是 `groot.vla.common` 的核心子模块，汇集了项目所需的所有工具函数。通过 `__init__.py` 统一导出三个子包的所有公开符号：

```python
from .data_structure import *
from .io import *
from .misc import *
```

这意味着用户可以直接从 `groot.vla.common.utils` 导入任何工具函数，例如：

```python
from groot.vla.common.utils import f_join, json_load, any_stack, is_sequence
```

## 子模块分类

### [io/ - 输入/输出工具](io/index.md)

处理各种数据格式的读写和配置管理：

| 文件 | 功能 |
|------|------|
| [config_utils.py](io/config_utils.md) | Hydra/OmegaConf 配置管理与类实例化 |
| [file_utils.py](io/file_utils.md) | 文件系统操作（路径、目录、读写） |
| [hdf5_utils.py](io/hdf5_utils.md) | HDF5 格式的递归读写与比较 |
| [json_utils.py](io/json_utils.md) | JSON/YAML/JSONL 读写 |
| [print_utils.py](io/print_utils.md) | 打印格式化、输出重定向、日志过滤 |
| [termcolor.py](io/termcolor.md) | 终端 ANSI 彩色文本 |

### [misc/ - 杂项工具](misc/index.md)

函数式编程、深度学习辅助和多媒体处理：

| 文件 | 功能 |
|------|------|
| [array_tensor_utils.py](misc/array_tensor_utils.md) | NumPy/PyTorch 通用数组操作 |
| [functional_utils.py](misc/functional_utils.md) | 元装饰器、递归函数、类注册 |
| [image_utils.py](misc/image_utils.md) | 图像格式转换、预处理、显示 |
| [misc_utils.py](misc/misc_utils.md) | 模式匹配、事件触发、嵌套访问 |
| [torch_utils.py](misc/torch_utils.md) | PyTorch 模型管理、训练工具 |
| [video_utils.py](misc/video_utils.md) | 多后端视频帧提取 |

### [data_structure/ - 数据结构工具](data_structure/index.md)

嵌套结构操作和形状计算：

| 文件 | 功能 |
|------|------|
| [shape_utils.py](data_structure/shape_utils.md) | 卷积/池化层输出形状推断 |
| [tree_utils.py](data_structure/tree_utils.md) | 嵌套数据结构的遍历与操作 |

## 模块依赖关系

依赖关系自底向上：

```
tree_utils.py (最底层，被几乎所有模块依赖)
    ↑
functional_utils.py (元编程基础)
    ↑
┌───┼───────────────┐
│   │               │
file_utils.py  misc_utils.py  array_tensor_utils.py
│       │               │
├───────┤               │
│       │               │
config_utils.py  print_utils.py  torch_utils.py
│                       │
json_utils.py    image_utils.py
                        │
                 video_utils.py
```

`hdf5_utils.py` 和 `termcolor.py` 相对独立，不依赖项目内其他模块。
