# 模型配置目录索引

## 概述

`groot/vla/configs/model/` 包含 DreamZero VLA 模型的配置文件。配置采用分层组合式设计，将模型的各个组件（backbone、action head、transform）解耦为独立的 YAML 文件。

## 子目录

| 目录 | 说明 |
|------|------|
| [dreamzero/](dreamzero/index.md) | DreamZero 模型配置（VLA 主配置、action head、backbone、transform） |

## 配置组合关系

```
model/dreamzero/vla.yaml (模型入口)
  ├── backbone/identity.yaml
  ├── action_head/wan_flow_matching_action_tf.yaml
  └── transform/dreamzero_cotrain.yaml
       └── base.yaml
```
