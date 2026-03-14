# `base_action_head.py` -- 动作头抽象基类

## 文件概述

本文件定义了 `ActionHead` 抽象基类，它是所有动作预测头（action head）的统一接口。ActionHead 接收 backbone 的输出特征，结合动作相关的输入（如历史动作、状态等），输出动作预测或训练损失。这是 VLA 架构中"感知-决策"分离设计的关键抽象层。

## 关键代码解析

```python
class ActionHead(ABC, nn.Module):
    def __init__(self):
        super(ActionHead, self).__init__()

    @abstractmethod
    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        pass

    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        num_action_samples: int = 1,
        inference_batch_size: int = 32,
    ) -> BatchFeature:
        return self.forward(backbone_output, action_input)

    def prepare_input(self, batch: dict) -> BatchFeature:
        pass

    def set_override_kwargs(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.config, key, value)
            setattr(self, key, value)
```

### 设计要点

1. **`forward`** 是抽象方法，子类必须实现。它接收 backbone 特征和动作输入，返回包含损失或动作预测的 `BatchFeature`。
2. **`get_action`** 用于推理时的动作预测，默认直接调用 `forward`，子类可以覆写以支持多次采样（`num_action_samples`）或分批推理（`inference_batch_size`）。
3. **`prepare_input`** 负责从原始数据字典中提取动作头所需的输入。
4. **`set_override_kwargs`** 允许在运行时动态修改配置参数。

## 核心类/函数表格

| 类/方法名 | 功能描述 |
|-----------|---------|
| `ActionHead` | 动作头抽象基类，定义标准接口 |
| `forward()` | 抽象方法，训练时的前向传播 |
| `get_action()` | 推理时获取动作预测 |
| `prepare_input()` | 从原始批次中提取动作头输入 |
| `set_override_kwargs()` | 运行时覆写配置参数 |

## 与其他模块的关系

- **被 `wan_flow_matching_action_tf.py` 继承**: DreamZero 的核心动作头继承此基类
- **被 `base_vla.py` 调用**: VLA 模型通过此接口调用动作头的 `forward`、`get_action` 和 `prepare_input`
- **使用 `BatchFeature`**: 统一使用 HuggingFace 的 BatchFeature 作为输入输出数据容器

## 总结

`ActionHead` 是一个轻量但关键的抽象类，它通过定义清晰的接口规范将动作预测逻辑与 backbone 特征提取解耦。这使得 DreamZero 可以灵活替换不同的动作预测策略（如 flow matching、diffusion policy 等），而不需要修改整体 VLA 架构。
