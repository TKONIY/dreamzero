# vla.yaml -- DreamZero VLA 模型配置

## 文件概述

`vla.yaml` 是 DreamZero VLA（Vision-Language-Action）模型的**核心模型定义配置文件**。它位于 `model/dreamzero/` 目录下，通过 Hydra 的 `defaults` 机制组合 backbone（骨干网络）、action_head（动作头）和 transform（数据变换）三个子模块的配置，定义了完整的 VLA 模型结构。

## 关键配置项解析

### 1. Hydra Defaults 组合

```yaml
defaults:
  - _self_
  - /model/dreamzero/backbone: identity
  - /model/dreamzero/action_head: wan_flow_matching_action_tf
  - /model/dreamzero/transform: dreamzero_cotrain
```

此处组合三个子配置：
- **backbone**：默认使用 `identity`（恒等骨干网络），即不使用额外的视觉/语言骨干编码器。
- **action_head**：使用基于 WAN（视频扩散模型）的 Flow Matching 动作头。
- **transform**：使用 DreamZero 联合训练的数据变换。

### 2. 模型实例化

```yaml
model:
  _target_: groot.vla.model.dreamzero.base_vla.VLA
  config:
    _target_: groot.vla.model.dreamzero.base_vla.VLAConfig
```

模型通过 Hydra 实例化为 `VLA` 类，其配置对象为 `VLAConfig`。`_recursive_: false` 避免 Hydra 自动递归实例化 `backbone_cfg` 和 `action_head_cfg`，让模型内部自行处理。

### 3. 跨文件参数引用

```yaml
hidden_size: ${backbone_hidden_size}
action_horizon: ${action_head_cfg.config.action_horizon}
action_dim: ${action_head_cfg.config.action_dim}
```

通过 Hydra 的变量插值，从 backbone 和 action_head 配置中自动获取关键维度参数，确保一致性。

## 配置参数表格

| 参数名 | 值/来源 | 说明 |
|--------|---------|------|
| `_target_` | `groot.vla.model.dreamzero.base_vla.VLA` | 模型类 |
| `_convert_` | `object` | Hydra 转换模式，确保配置可序列化 |
| `model_dtype` | `float32` | 模型计算精度 |
| `hidden_size` | `${backbone_hidden_size}` | 隐藏层维度，来自 backbone 配置 |
| `action_horizon` | `${action_head_cfg.config.action_horizon}` | 动作预测时间步数，来自 action_head |
| `action_dim` | `${action_head_cfg.config.action_dim}` | 动作维度，来自 action_head |
| `backbone_cfg` | `${backbone_cfg}` | 骨干网络完整配置 |
| `action_head_cfg` | `${action_head_cfg}` | 动作头完整配置 |

## 与其他配置/模块的关系

- **被引用**：由 `conf.yaml` 的 `defaults` 中 `model: dreamzero/vla` 引入。
- **引入的子配置**：
  - `backbone/identity.yaml`：定义 Identity 骨干网络（无特征提取）。
  - `action_head/wan_flow_matching_action_tf.yaml`：定义 WAN Flow Matching 动作头。
  - `transform/dreamzero_cotrain.yaml`：定义联合训练数据变换和数据整理器。
- **运行时关系**：`VLA` 类在初始化时将 `backbone_cfg` 和 `action_head_cfg` 分别实例化为对应的 backbone 和 action_head 模块。

## 总结

`vla.yaml` 是 DreamZero 模型架构的枢纽配置。它采用模块化设计，通过 Hydra defaults 组合 backbone、action_head 和 transform 三个独立可替换的子模块。当前默认配置使用 Identity backbone（因为 DreamZero 的视觉编码由 WAN 扩散模型内部完成）和基于 WAN 视频生成模型的 Flow Matching 动作头，体现了 DreamZero "以视频生成模型为核心" 的设计理念。
