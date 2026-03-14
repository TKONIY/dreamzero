# 动作头（Action Head）配置索引

## 概述

`action_head/` 目录包含 DreamZero 动作头模块的配置文件。动作头是模型中负责从视觉-语言条件生成机器人动作序列的核心组件。

## 文件索引

- [wan_flow_matching_action_tf.yaml](wan_flow_matching_action_tf.md) -- 基于 WAN 视频扩散模型的 Flow Matching 动作头

## 模块说明

`WANPolicyHead` 是 DreamZero 的核心创新模块，它将 WAN（万相）视频生成模型改造为同时预测视频帧和机器人动作的统一模型。关键特性包括：

- **Flow Matching**：使用流匹配（而非传统 DDPM）进行扩散建模，推理时仅需 4 步即可生成。
- **因果分块注意力**：支持高效的自回归推理。
- **LoRA 微调**：仅在注意力和 FFN 层插入 rank=4 的低秩适配器。
- **多组件集成**：包含文本编码器、图像编码器、VAE、VL 融合层和 40 层 DiT。
