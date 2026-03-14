# DreamZero 模型配置索引

## 概述

`model/dreamzero/` 目录定义了 DreamZero VLA 模型的完整架构配置。模型采用模块化设计，由三个可替换的子模块组成：骨干网络（backbone）、动作头（action_head）和数据变换（transform）。

## 文件索引

### 模型入口
- [vla.yaml](vla.md) -- VLA 模型主配置，组合三个子模块

### 子模块目录
- [action_head/](action_head/index.md) -- 动作头配置（基于 WAN 视频扩散模型）
- [backbone/](backbone/index.md) -- 骨干网络配置（Identity 占位）
- [transform/](transform/index.md) -- 数据变换与体型映射配置

## 架构概览

```
VLA (base_vla.VLA)
├── backbone: IdentityBackbone        # 无额外视觉编码
└── action_head: WANPolicyHead        # WAN Flow Matching 动作头
    ├── WanTextEncoder                # 文本编码器
    ├── WanImageEncoder               # 图像编码器（I2V 条件）
    ├── WanVideoVAE                   # 视频 VAE
    ├── SelfAttentionTransformer      # VL 自注意力融合
    └── CausalWanModel                # 因果 WAN DiT（40 层）
```

## 设计特点

DreamZero 的核心创新在于将动作预测嵌入视频扩散过程中，因此：
- **backbone 为 Identity**：不需要额外的视觉编码器，视觉理解由 WAN 内部完成。
- **action_head 包含完整的生成模型**：包括文本编码器、图像编码器、VAE 和 40 层 DiT。
- **使用 LoRA 微调**：在 WAN 预训练权重基础上，仅通过低秩适配器进行微调。
