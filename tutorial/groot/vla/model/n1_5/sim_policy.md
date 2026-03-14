# `sim_policy.py` -- 仿真推理策略管理器

## 文件概述

本文件实现了 DreamZero 项目在仿真环境中进行推理时的策略封装。核心功能包括：

1. **ModelManager**: 管理多模型的 GPU 显存调度，保持 VLM 常驻显存，按需加载/卸载 action_head 的组件（text_encoder、vae 等）。
2. **BaseGrootSimPolicy**: 基于 Tianshou 框架的仿真策略基类，定义了推理所需的公共接口。
3. **GrootSimPolicy**: 完整的仿真推理策略，负责模型加载、数据变换、动作预测以及多模型协同调度。

该文件是连接训练好的 VLA 模型和仿真环境的桥梁，使得训练完成的模型可以直接在机器人仿真器中执行推理。

## 关键代码解析

### ModelManager -- 显存管理器

```python
class ModelManager:
    def __init__(self):
        self.active_components = None
        self.models = {}
        self.vlm_policy = None
        self.action_head_policy = None

    def register_model(self, name: str, policy_instance):
        self.models[name] = policy_instance
        if name == "vlm":
            self.vlm_policy = policy_instance
            self.load_vlm_model()
        elif name == "action_head":
            self.action_head_policy = policy_instance
            self.enable_action_head_vram_management()
            self.offload_action_head_components()
```

ModelManager 采用"VLM 常驻 + action_head 按需加载"的策略。VLM 模型体积较小且需频繁调用，因此始终保持在 GPU 上；action_head 中的 text_encoder 和 VAE 组件体积大但调用频率低，因此仅在需要时加载到 GPU，用完后卸载回 CPU，从而在单 GPU 上实现大模型推理。

### BaseGrootSimPolicy -- 策略基类

```python
class BaseGrootSimPolicy(BaseTianshouPolicy):
    def __init__(self, embodiment_tag: EmbodimentTag, model_path: str, device: int | str):
        super().__init__()
        self.embodiment_tag = embodiment_tag
        self.model_path = model_path
        self.device = device
```

基类继承自 Tianshou 的 `BasePolicy`，定义了 `embodiment_tag`（机器人形态标签）、`model_path`（模型路径）和 `device`（推理设备）三个核心属性，并提供了 `video_delta_indices`、`state_delta_indices` 等属性接口供子类覆写。

### GrootSimPolicy -- 完整推理策略

GrootSimPolicy 是核心推理类，主要流程：

1. **初始化阶段**：通过 Hydra 配置加载 VLA 模型和数据变换器
2. **forward 方法**：处理来自仿真环境的观测数据，经过变换后送入 VLA 模型推理，获取动作预测
3. **多模型推理**：支持 VLM + action_head 的分步推理流程

## 核心类/函数表格

| 类/函数名 | 功能描述 |
|-----------|---------|
| `ModelManager` | GPU 显存管理器，协调多模型的加载/卸载 |
| `ModelManager.register_model()` | 注册策略实例，自动管理加载状态 |
| `ModelManager.activate_model()` | 激活指定模型的 GPU 计算 |
| `ModelManager.offload_action_head_components()` | 将 action_head 组件卸载到 CPU |
| `BaseGrootSimPolicy` | 仿真策略基类，定义公共接口 |
| `GrootSimPolicy` | 完整的仿真推理策略实现 |

## 与其他模块的关系

- **依赖 `groot.vla.data.schema`**: 使用 `EmbodimentTag` 和 `DatasetMetadata` 标识机器人形态
- **依赖 `groot.vla.data.transform`**: 使用 `ComposedModalityTransform` 进行数据预处理
- **依赖 `base_vla.py`**: 加载的模型是 VLA 类实例
- **依赖 `wan_flow_matching_action_tf.py`**: action_head 的推理入口
- **被仿真环境调用**: 作为 Tianshou 策略集成到仿真循环中

## 总结

`sim_policy.py` 是 DreamZero 推理链的最上层封装。它通过 ModelManager 解决了大模型在有限显存下的推理问题，通过 BaseGrootSimPolicy 提供了标准化的策略接口，通过 GrootSimPolicy 完成了从环境观测到动作预测的完整流程。这一设计使得同一份训练好的模型可以灵活部署到不同的仿真环境中。
