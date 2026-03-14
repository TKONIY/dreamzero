# `dreamzero/transform/` -- 数据变换模块

## 模块概述

实现 DreamZero 的数据预处理管线，将多机器人形态、多视角的原始数据统一转换为模型可消费的格式。

## 文件列表

| 文件 | 描述 |
|------|------|
| [`dreamzero_cotrain.py`](dreamzero_cotrain.md) | 核心数据变换：多视角拼接、语言预处理、动作填充、多训练模式支持 |
| [`common.py`](common.md) | 通用文本处理工具：`formalize_language` 函数 |

## `__init__.py`

`__init__.py` 文件为空。
