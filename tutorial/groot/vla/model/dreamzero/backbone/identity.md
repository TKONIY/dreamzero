# `identity.py` -- 恒等 Backbone（Identity Backbone）

## 文件概述

本文件实现了 `IdentityBackbone`，一个不执行任何实际计算的"透传"backbone。它的作用是在不依赖任何视觉语言模型（VLM）的情况下，使 action_head 可以作为独立模型进行训练和推理。这在 DreamZero 的架构中尤为重要，因为 DreamZero 的 action_head（基于 Wan2.1 DiT）本身已经内置了视觉编码能力。

## 关键代码解析

```python
class IdentityBackbone(Backbone):
    def set_trainable_parameters(self, **kwargs):
        return  # 无参数可训练

    def forward(self, backbone_input: BatchFeature) -> BatchFeature:
        backbone_input_first_value = next(iter(backbone_input.values()))
        B = backbone_input_first_value.shape[0]
        backbone_features = torch.empty(
            B, 1, 0, dtype=torch.float32, device=backbone_input_first_value.device
        )
        output_dict = {"backbone_features": backbone_features}
        return BatchFeature(data=output_dict)
```

`forward` 方法输出一个形状为 `(B, 1, 0)` 的空张量作为 `backbone_features`。这里维度为 0 的最后一维确保 action_head 不会从 backbone 获取任何实质特征，从而保证 action_head 的独立性。

```python
def prepare_input(self, batch: dict) -> BatchFeature:
    if "action" in batch:
        return BatchFeature(data={"action": batch["action"]})
    elif "state" in batch:
        return BatchFeature(data={"state": batch["state"]})
    elif "video" in batch:
        video_tensor = torch.from_numpy(batch["video"])
        return BatchFeature(data={"video": video_tensor})
    else:
        return BatchFeature(data=batch)
```

`prepare_input` 按优先级选择第一个可用的输入键：训练时用 `action`，推理时用 `state` 或 `video`。这些输入仅用于推断 batch_size，不参与计算。

## 核心类/函数表格

| 类/方法名 | 功能描述 |
|-----------|---------|
| `IdentityBackbone` | 恒等 backbone，不做任何特征提取 |
| `forward()` | 返回空的 backbone_features |
| `prepare_input()` | 从批次中提取任意键以推断 batch_size |
| `set_trainable_parameters()` | 空操作，无可训练参数 |

## 与其他模块的关系

- **继承 `base_backbone.py`**: 实现 `Backbone` 接口
- **被 `base_vla.py` 的 VLA 使用**: 当配置中指定使用 IdentityBackbone 时
- **与 `wan_flow_matching_action_tf.py` 配合**: action_head 自身包含完整的视觉处理能力，不需要外部 backbone

## 总结

`IdentityBackbone` 是 DreamZero 架构灵活性的体现。由于 DreamZero 的核心创新在于将 Wan2.1 视频生成模型作为 action_head（其内部已包含完整的视觉/语言编码能力），因此不需要额外的 backbone 来提取特征。IdentityBackbone 通过提供一个空的 backbone_features，满足了 VLA 接口要求，同时允许 action_head 完全自主地处理所有输入模态。
