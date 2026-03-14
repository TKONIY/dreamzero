# `base_vla.py` -- VLA 模型基类（视觉-语言-动作架构）

## 文件概述

本文件定义了 DreamZero 的核心模型架构 -- VLA (Vision-Language-Action)。VLA 采用"backbone + action_head"的双模块设计：backbone 负责处理视觉和语言输入，提取高级特征；action_head 负责基于这些特征预测机器人动作。文件还提供了多种模型加载方式（预训练、LoRA 微调等）以及联合视频-动作生成的推理接口。

## 关键代码解析

### VLAConfig -- 配置类

```python
@dataclass
class VLAConfig(PretrainedConfig):
    model_type = "vla"
    backbone_cfg: PretrainedConfig = field(default=None)
    action_head_cfg: PretrainedConfig = field(default=None)
    action_horizon: int = field(default=None)
    action_dim: int = field(default=None)
    compute_dtype: str = field(default="float32")
```

VLAConfig 继承自 HuggingFace 的 `PretrainedConfig`，包含 backbone 和 action_head 的配置、动作预测时间步长（action_horizon）以及动作维度等关键参数。

### VLA -- 核心模型类

```python
class VLA(PreTrainedModel):
    def __init__(self, config: VLAConfig):
        super().__init__(config)
        self.backbone = instantiate(config.backbone_cfg)      # 通过 Hydra 实例化 backbone
        self.action_head = instantiate(config.action_head_cfg) # 通过 Hydra 实例化 action_head
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim

    def forward(self, inputs: dict) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        return action_head_outputs
```

VLA 的前向传播流程：
1. `prepare_input`: 分别为 backbone 和 action_head 准备输入
2. `self.backbone(backbone_inputs)`: 提取视觉/语言特征
3. `self.action_head(backbone_outputs, action_inputs)`: 基于特征预测动作

### 多种推理模式

```python
def get_action(self, inputs):          # 标准动作预测
def joint_video_action(self, inputs):  # 联合视频+动作生成
def lazy_joint_video_action(self, inputs):  # 惰性联合生成（节省显存）
def lazy_joint_video_action_causal(self, inputs, latent_video=None):  # 因果式联合生成
def gt_video_action_pred(self, inputs): # 使用真实视频预测动作
def get_video(self, inputs):           # 仅生成视频
```

DreamZero 支持多种推理模式，其中最重要的是 `lazy_joint_video_action_causal`，它实现了因果式的视频-动作联合生成，支持自回归地逐帧生成视频和动作。

### 模型加载方法

```python
@classmethod
def from_pretrained_for_tuning(cls, pretrained_model_name_or_path, config=None, ...):
    model = cls(config)
    # 支持分片 safetensors 加载
    # 支持 LoRA 权重加载
    return model

@classmethod
def load_lora(cls, pretrained_model_name_or_path):
    # 加载带 LoRA 适配器的完整模型
    # 处理 base_layer 键名映射

def load_lora_weight(self, pretrained_model_name_or_path):
    # 仅加载 LoRA 权重到已有模型
    # 处理键名从 action_head.model -> action_head.model.base_model.model 的映射
```

### CotrainVLA -- 联合训练变体

```python
class CotrainVLA(VLA):
    def forward(self, inputs: dict) -> BatchFeature:
        if "cotrain" in inputs and inputs["cotrain"]:
            return self.backbone.cotrain(inputs)
        return super().forward(inputs)
```

CotrainVLA 支持联合训练模式，当输入标记为 cotrain 时，直接使用 backbone 的联合训练路径。

## 核心类/函数表格

| 类/函数名 | 功能描述 |
|-----------|---------|
| `VLAConfig` | VLA 模型配置数据类 |
| `VLA` | 核心 VLA 模型，组合 backbone + action_head |
| `VLA.forward()` | 训练时的前向传播 |
| `VLA.get_action()` | 推理时获取动作预测 |
| `VLA.joint_video_action()` | 联合视频-动作生成 |
| `VLA.lazy_joint_video_action_causal()` | 因果式联合生成 |
| `VLA.from_pretrained_for_tuning()` | 加载预训练模型用于微调 |
| `VLA.load_lora()` | 加载 LoRA 适配器权重 |
| `CotrainVLA` | 支持联合训练的 VLA 变体 |
| `create_vla_with_pretrained_action_head()` | 用预训练 action_head 创建 VLA |

## 与其他模块的关系

- **使用 `backbone/` 模块**: backbone 子模块（如 `IdentityBackbone`）处理感知输入
- **使用 `action_head/` 模块**: action_head 子模块（如 `WanFlowMatchingActionTF`）预测动作
- **被 `sim_policy.py` 调用**: 仿真策略加载和使用 VLA 模型
- **被 `dreamzero_cotrain.py` 数据变换支持**: 数据变换为 VLA 准备训练/推理输入
- **注册到 HuggingFace**: 通过 `AutoConfig.register` 和 `AutoModel.register` 注册

## 总结

`base_vla.py` 是 DreamZero 架构的中枢，它将视觉感知（backbone）和动作决策（action_head）组合为统一的端到端模型。通过支持多种推理模式（标准推理、联合生成、因果生成）和多种权重加载方式（全量加载、LoRA 微调），VLA 模型兼具灵活性和实用性。
