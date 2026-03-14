# groot.vla.common 模块

## 模块概述

`groot.vla.common` 是 DreamZero 项目的公共工具模块，为整个 VLA（Vision-Language-Action）系统提供基础设施支持。该模块本身的 `__init__.py` 为空，所有功能通过子模块 `utils` 提供。

## 目录结构

```
groot/vla/common/
├── __init__.py              # 空文件
└── utils/                   # 工具函数集合
    ├── __init__.py           # 统一导出所有子模块
    ├── io/                   # 输入/输出工具
    │   ├── config_utils.py   # 配置管理（Hydra/OmegaConf）
    │   ├── file_utils.py     # 文件系统操作
    │   ├── hdf5_utils.py     # HDF5 数据读写
    │   ├── json_utils.py     # JSON/YAML 读写
    │   ├── print_utils.py    # 打印与日志工具
    │   └── termcolor.py      # 终端彩色输出
    ├── misc/                 # 杂项工具
    │   ├── array_tensor_utils.py  # NumPy/PyTorch 通用操作
    │   ├── functional_utils.py    # 函数式编程与元编程
    │   ├── image_utils.py         # 图像处理
    │   ├── misc_utils.py          # 通用工具函数
    │   ├── torch_utils.py         # PyTorch 专用工具
    │   └── video_utils.py         # 视频处理
    └── data_structure/       # 数据结构工具
        ├── shape_utils.py    # 形状推断
        └── tree_utils.py     # 嵌套结构操作
```

## 子模块导航

- [utils/ - 工具函数集合](utils/index.md)
  - [io/ - 输入/输出工具](utils/io/index.md)
  - [misc/ - 杂项工具](utils/misc/index.md)
  - [data_structure/ - 数据结构工具](utils/data_structure/index.md)

## 设计理念

`common` 模块遵循以下设计原则：

1. **扁平化导入**：通过各级 `__init__.py` 的 `from .xxx import *`，用户可以从 `groot.vla.common.utils` 直接导入任何工具函数，无需记忆具体文件位置。

2. **最小依赖**：底层模块（如 `tree_utils.py`、`functional_utils.py`）不依赖上层模块，形成清晰的依赖层次。

3. **类型无关设计**：核心操作函数（如 `any_stack`、`any_zeros_like`）同时支持 NumPy 和 PyTorch，降低调用者的心智负担。

4. **递归结构支持**：通过 `@make_recursive_func` 装饰器和 `dm-tree` 库，大量函数能透明地处理任意深度的嵌套数据结构。
