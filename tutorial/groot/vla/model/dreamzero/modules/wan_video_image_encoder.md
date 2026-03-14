# `wan_video_image_encoder.py` -- Wan 图像编码器（CLIP ViT）

## 文件概述

本文件实现了 Wan2.1 使用的图像编码器，基于 CLIP (Contrastive Language-Image Pre-training) 的 Vision Transformer (ViT-H/14) 架构。在 DreamZero 中，图像编码器用于提取视频第一帧的视觉特征作为 Image-to-Video (I2V) 生成的条件。

文件包含了完整的 CLIP 模型实现：视觉 Transformer、文本 Transformer（XLMRoberta）、以及多种预训练模型加载方式。

## 关键代码解析

### VisionTransformer -- 视觉 Transformer

```python
class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, dim=768, ...):
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(gain * torch.randn(1, num_patches + 1, dim))
        self.transformer = nn.Sequential(*[AttentionBlock(...) for _ in range(num_layers)])

    def forward(self, x, use_31_block=False):
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        x = torch.cat([self.cls_embedding, x], dim=1)
        x = x + self.pos_embedding
        if use_31_block:
            x = self.transformer[:-1](x)  # 使用前31层（不含最后一层）
        else:
            x = self.transformer(x)
        return x
```

注意 `use_31_block=True` 时只使用前 31 层，输出中间层特征而非最终特征。这是 Wan2.1 I2V 模型的设计选择，使用中间特征能保留更多空间信息。

### WanImageEncoder -- 图像编码器封装

```python
class WanImageEncoder(torch.nn.Module):
    def __init__(self, image_encoder_pretrained_path=None):
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False, return_transforms=True)

    def encode_image(self, videos):
        size = (self.model.image_size,) * 2
        videos = torch.cat([
            F.interpolate(u, size=size, mode='bicubic', align_corners=False)
            for u in videos
        ])
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))
        dtype = next(iter(self.model.visual.parameters())).dtype
        out = self.model.visual(videos.to(dtype), use_31_block=True)
        return out.clone()
```

`encode_image` 的处理流程：
1. 将视频帧缩放到 CLIP 输入尺寸（224x224）
2. 应用 CLIP 标准化（均值/标准差归一化）
3. 通过 ViT-H/14 的前 31 层提取特征
4. 输出形状: `[B, 257, 1280]`（257 = 1 CLS token + 256 patch tokens，1280 为 ViT-H 维度）

### XLMRoberta -- 多语言文本编码器

文件还包含了 XLMRoberta 的实现，但在 DreamZero 中主要使用 `WanTextEncoder`（基于 UMT5），XLMRoberta 仅作为 CLIP 文本分支的一部分。

## 核心类/函数表格

| 类/函数名 | 功能描述 |
|-----------|---------|
| `VisionTransformer` | CLIP ViT 视觉编码器 |
| `CLIP` | 完整的 CLIP 模型（视觉+文本） |
| `XLMRobertaCLIP` | 使用 XLMRoberta 作为文本编码器的 CLIP |
| `WanImageEncoder` | Wan 图像编码器封装 |
| `encode_image()` | 提取图像的 CLIP 特征 |
| `clip_xlm_roberta_vit_h_14()` | 创建 ViT-H/14 + XLMRoberta 的 CLIP 模型 |
| `WanImageEncoderStateDictConverter` | 权重格式转换器 |

## 与其他模块的关系

- **被 `wan_flow_matching_action_tf.py` 使用**: 提供 `clip_feature` 条件
- **输出供 `wan_video_dit.py` 消费**: `img_emb` MLP 将 1280 维映射到 DiT 的维度
- **使用 `wan_video_dit.py` 的 `flash_attention`**: 注意力计算共享实现

## 总结

`wan_video_image_encoder.py` 实现了基于 CLIP ViT-H/14 的图像编码器，为 DreamZero 的 I2V 生成提供视觉条件。通过提取第一帧的 CLIP 特征（257 token x 1280 维），DiT 模型可以生成与初始帧视觉风格和内容一致的后续视频帧。
