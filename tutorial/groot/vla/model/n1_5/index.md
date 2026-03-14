# `n1_5/` -- N1.5 基础框架模块

## 模块概述

N1.5 模块提供了 DreamZero 项目继承自 GROOT N1.5 的基础组件。这些组件定义了核心接口和通用模块，被 DreamZero 的核心实现所使用。

## 子模块

| 文件/目录 | 描述 |
|-----------|------|
| [`sim_policy.py`](sim_policy.md) | 仿真推理策略管理器，包括 ModelManager 显存调度和 GrootSimPolicy 推理策略 |
| [`action_head/`](action_head/index.md) | 动作头接口定义（`ActionHead` 抽象基类） |
| [`modules/`](modules/index.md) | 基础神经网络模块（`ActionEncoder` 动作编码器） |

## 设计理念

N1.5 层定义了"做什么"（接口），DreamZero 层实现了"怎么做"（具体实现）。这种分层设计使得 DreamZero 可以灵活替换底层实现，同时保持与 GROOT 生态系统的兼容性。

## `__init__.py`

`__init__.py` 文件为空。
