# identity.yaml -- Identity 骨干网络配置

## 文件概述

`identity.yaml` 定义了一个**恒等（Identity）骨干网络**配置。在 DreamZero 架构中，由于视觉编码功能已由 WAN 视频扩散模型内部的 VAE 和图像编码器承担，因此不需要额外的独立骨干网络。此配置通过一个 `IdentityBackbone` 类实现了"无操作"的骨干网络占位。

## 关键配置项解析

### 1. backbone_hidden_size

```yaml
backbone_hidden_size: 0
```

将骨干网络的隐藏维度设为 0，表示不产生任何骨干特征输出。此值会被 `vla.yaml` 引用作为模型的 `hidden_size`。

### 2. backbone_cfg

```yaml
backbone_cfg:
  _target_: groot.vla.model.dreamzero.backbone.identity.IdentityBackbone
```

指向 `IdentityBackbone` 类，该类是一个不执行任何计算的空骨干网络。

### 3. 禁用特征投影

```yaml
action_head_cfg:
  config:
    backbone_features_projector_cfg: null
```

由于没有骨干特征输出，将 action_head 中的骨干特征投影器设为 `null`，避免不必要的投影层。

## 配置参数表格

| 参数名 | 值 | 说明 |
|--------|------|------|
| `backbone_hidden_size` | `0` | 骨干网络输出维度（0 表示无输出） |
| `backbone_cfg._target_` | `groot.vla.model.dreamzero.backbone.identity.IdentityBackbone` | 恒等骨干网络类 |
| `backbone_features_projector_cfg` | `null` | 禁用骨干特征投影器 |

## 与其他配置/模块的关系

- **被引用**：由 `model/dreamzero/vla.yaml` 的 `defaults` 中 `/model/dreamzero/backbone: identity` 引入。
- **影响 vla.yaml**：`backbone_hidden_size` 被 `vla.yaml` 中的 `hidden_size: ${backbone_hidden_size}` 引用。
- **影响 action_head**：将 `backbone_features_projector_cfg` 设为 `null`，告诉动作头不需要处理骨干特征。

## 总结

`identity.yaml` 是一个极简的骨干网络配置，本质上是一个"空操作"占位符。这反映了 DreamZero 的设计选择：不使用传统的独立视觉骨干网络（如 ViT、ResNet），而是将视觉理解能力完全内置于 WAN 视频扩散模型中。如果未来需要在 DreamZero 中添加额外的视觉编码器，可以创建新的 backbone 配置文件替换此文件。
