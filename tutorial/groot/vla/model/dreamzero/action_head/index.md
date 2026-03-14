# `dreamzero/action_head/` -- DreamZero 动作头

## 模块概述

包含 DreamZero 的核心动作头实现，基于 Wan2.1 视频生成模型的 Flow Matching 动作预测。

## 文件列表

| 文件 | 描述 |
|------|------|
| [`wan_flow_matching_action_tf.py`](wan_flow_matching_action_tf.md) | 核心动作头：集成 DiT、文本编码器、图像编码器、VAE，实现联合视频-动作生成 |

这是 DreamZero 系统中最核心、代码量最大的单个文件，包含了完整的训练和推理逻辑。

## `__init__.py`

`__init__.py` 文件为空。
