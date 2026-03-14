# `groot/vla/model/` -- DreamZero 模型模块总览

## 模块概述

`groot/vla/model/` 是 DreamZero 项目的核心模型层，包含了从视觉感知到动作预测的完整 VLA (Vision-Language-Action) 架构。模块按功能组织为两个子包：

## 子模块结构

### [`n1_5/`](n1_5/index.md) -- N1.5 基础框架
继承自 GROOT N1.5 的基础组件，提供了策略接口、动作头抽象基类和动作编码器。

- [`sim_policy.py`](n1_5/sim_policy.md) -- 仿真推理策略管理器
- [`action_head/`](n1_5/action_head/index.md) -- 动作头接口定义
- [`modules/`](n1_5/modules/index.md) -- 基础神经网络模块

### [`dreamzero/`](dreamzero/index.md) -- DreamZero 核心实现
DreamZero 的全部核心代码，基于 Wan2.1 视频生成模型实现了联合世界模型+策略模型。

- [`base_vla.py`](dreamzero/base_vla.md) -- VLA 模型基类
- [`action_head/`](dreamzero/action_head/index.md) -- Flow Matching 动作头
- [`backbone/`](dreamzero/backbone/index.md) -- Backbone 模块
- [`transform/`](dreamzero/transform/index.md) -- 数据变换
- [`modules/`](dreamzero/modules/index.md) -- 核心神经网络组件

## 架构总览

```
输入: 视频 + 语言指令 + 机器人状态
  |
  v
[DreamTransform] -- 数据预处理（多视角拼接、分词、维度填充）
  |
  v
[VLA] = [IdentityBackbone] + [WanFlowMatchingActionTF]
  |                               |
  | (空特征)                       |-- [WanTextEncoder] -- 文本编码
  |                               |-- [WanImageEncoder] -- 图像编码
  |                               |-- [VideoVAE] -- 视频编解码
  |                               |-- [WanModel/DiT] -- 联合去噪
  |                               |     |-- 视频 token + 动作 token
  |                               |     |-- 因果注意力 + 交叉注意力
  |                               |-- [FlowMatchScheduler] -- 噪声调度
  |
  v
输出: 动作预测 + 视频预测
```

## `__init__.py`

`__init__.py` 文件为空，仅用于将目录标记为 Python 包。
