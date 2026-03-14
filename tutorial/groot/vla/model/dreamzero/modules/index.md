# `dreamzero/modules/` -- 核心神经网络组件

## 模块概述

包含 DreamZero 所有底层神经网络组件，涵盖 DiT 模型、编码器、VAE、注意力机制、噪声调度和显存管理。

## 文件列表

### DiT 模型

| 文件 | 描述 |
|------|------|
| [`wan_video_dit.py`](wan_video_dit.md) | 标准 Wan DiT 模型（早期版本，兼容旧权重） |
| [`wan_video_dit_action_casual_chunk.py`](wan_video_dit_action_casual_chunk.md) | 因果式动作-视频联合 DiT（核心版本，支持动作 token + 因果注意力） |

### 编码器

| 文件 | 描述 |
|------|------|
| [`wan_video_text_encoder.py`](wan_video_text_encoder.md) | UMT5 文本编码器 |
| [`wan_video_image_encoder.py`](wan_video_image_encoder.md) | CLIP ViT-H/14 图像编码器 |
| [`wan_video_vae.py`](wan_video_vae.md) | 3D 视频 VAE（编码器 + 解码器） |

### 注意力机制

| 文件 | 描述 |
|------|------|
| [`attention.py`](attention.md) | 基础 Flash Attention 函数 |
| [`wan2_1_attention.py`](wan2_1_attention.md) | 面向对象的注意力模块（多后端） |
| [`wan2_1_submodule.py`](wan2_1_submodule.md) | Wan2.1 DiT 子模块集合 |
| [`cudnn_attention.py`](cudnn_attention.md) | cuDNN 融合注意力（Transformer Engine） |

### 调度器

| 文件 | 描述 |
|------|------|
| [`flow_match_scheduler.py`](flow_match_scheduler.md) | Flow Matching 基础调度器（欧拉法） |
| [`flow_unipc_multistep_scheduler.py`](flow_unipc_multistep_scheduler.md) | UniPC 多步高阶调度器 |

### 基础设施

| 文件 | 描述 |
|------|------|
| [`wan_video_camera_controller.py`](wan_video_camera_controller.md) | 相机控制模块（Plucker 坐标） |
| [`vram_management.py`](vram_management.md) | GPU 显存管理（权重按需加载 + FP8 + LoRA） |
| [`utils.py`](utils.md) | 通用工具函数（权重加载、Meta 设备、哈希） |

## 组件交互关系

```
[WanTextEncoder] ----\
                      \
[WanImageEncoder] -----+---> [WanModel/DiT] ---> 去噪输出
                      /        ^    ^
[VideoVAE.encode] ---/         |    |
                     [FlowMatchScheduler]  [AttentionModule]
                               |
[VideoVAE.decode] <--- 去噪后的潜在表示
```

## `__init__.py`

`__init__.py` 文件为空。
