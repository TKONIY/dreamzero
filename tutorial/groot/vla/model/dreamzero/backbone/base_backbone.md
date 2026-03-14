# `base_backbone.py` -- Backbone 抽象基类

## 文件概述

本文件定义了 `Backbone` 抽象基类，是 VLA 架构中感知模块的统一接口。Backbone 负责处理视觉和/或语言输入，输出 `backbone_features` 供下游的 action_head 使用。

## 关键代码解析

```python
class Backbone(ABC, nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

    @abstractmethod
    def forward(self, backbone_input: BatchFeature) -> BatchFeature:
        pass

    def prepare_input(self, batch: dict) -> BatchFeature:
        pass
```

接口设计极简：
- `forward`: 抽象方法，接收 `BatchFeature` 输入，输出包含 `backbone_features` 键的 `BatchFeature`
- `prepare_input`: 从原始数据字典中提取 backbone 所需的输入

## 核心类/函数表格

| 类/方法名 | 功能描述 |
|-----------|---------|
| `Backbone` | Backbone 抽象基类 |
| `forward()` | 抽象前向传播方法 |
| `prepare_input()` | 输入预处理方法 |

## 与其他模块的关系

- **被 `identity.py` 继承**: `IdentityBackbone` 实现了一个透传 backbone
- **被 `base_vla.py` 调用**: VLA 模型的 `forward` 方法调用 `self.backbone(backbone_inputs)`

## 总结

`Backbone` 定义了感知模块的最小接口，使得 VLA 架构可以灵活搭配不同的视觉语言模型（VLM）或使用 `IdentityBackbone` 进行独立的 action_head 训练。
