# dreamzero_cotrain.yaml -- DreamZero 联合训练数据变换配置

## 文件概述

`dreamzero_cotrain.yaml` 定义了 DreamZero 模型的**数据变换（Transform）和数据整理器（Data Collator）** 配置。该文件继承自 `base.yaml`（机器人体型映射），并定义了两个关键组件：
1. **DefaultDataCollator**：负责将多个样本整理为训练批次。
2. **DreamTransform**：模型特定的数据变换，将原始数据样本转换为模型可接受的输入格式。

## 关键配置项解析

### 1. Hydra Defaults 继承

```yaml
defaults:
  - base
  - _self_
```

继承 `base.yaml` 中的 `embodiment_tag_to_projector_index` 映射表，然后用当前文件的配置覆盖。

### 2. Tokenizer 配置

```yaml
tokenizer_path: "google/umt5-xxl"
max_length: 512
```

使用 Google 的 UMT5-XXL 分词器处理语言指令，最大 token 长度为 512。

### 3. Data Collator（数据整理器）

```yaml
data_collator:
  _target_: groot.vla.model.dreamzero.transform.dreamzero_cotrain.DefaultDataCollator
  tokenizer_path: ${tokenizer_path}
  max_length: ${max_length}
  num_views: ${num_views}
  embodiment_tag_mapping: ${embodiment_tag_to_projector_index}
```

`DefaultDataCollator` 负责：
- 使用指定的分词器对语言指令进行编码
- 根据视角数（`num_views`）处理多视角视频数据
- 根据体型映射为每个样本分配正确的投影器索引
- 将多个样本组成批次

### 4. 模型特定变换（Model-Specific Transform）

```yaml
model_specific_transform:
  _target_: groot.vla.model.dreamzero.transform.dreamzero_cotrain.DreamTransform
  default_instruction: "Perform the default behavior."
  language_dropout_prob: ${language_dropout_prob}
  always_use_default_instruction: false
```

`DreamTransform` 是在通用数据变换（如视频裁剪、归一化）之后应用的模型特定变换，负责：
- 将数据格式化为 DreamZero 模型期望的输入结构
- 处理语言指令（支持随机 dropout 和默认指令替换）
- 填充/截断状态和动作到统一维度（`max_state_dim`、`max_action_dim`）

### 5. 关键维度参数

```yaml
num_visual_tokens_per_frame: 16
max_state_dim: 44
max_action_dim: 32
language_dropout_prob: 0.0
```

## 配置参数表格

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `max_length` | `512` | 语言指令最大 token 长度 |
| `num_views` | `1` | 视频视角数 |
| `tokenizer_path` | `"google/umt5-xxl"` | 分词器路径（UMT5-XXL） |
| `num_visual_tokens_per_frame` | `16` | 每帧视觉 token 数 |
| `max_state_dim` | `44` | 最大状态维度 |
| `max_action_dim` | `32` | 最大动作维度 |
| `language_dropout_prob` | `0.0` | 语言指令随机丢弃概率 |
| `default_instruction` | `"Perform the default behavior."` | 默认语言指令 |
| `always_use_default_instruction` | `false` | 是否始终使用默认指令 |

## 与其他配置/模块的关系

- **继承自**：`base.yaml`（获取 `embodiment_tag_to_projector_index`）。
- **被引用**：由 `model/dreamzero/vla.yaml` 的 `defaults` 中 `/model/dreamzero/transform: dreamzero_cotrain` 引入。
- **在数据管道中的位置**：`model_specific_transform` 被插入到每个数据集的变换链末尾（参见数据配置中的 `- ${model_specific_transform}`）。
- **与 action_head 的关系**：`max_state_dim` 和 `max_action_dim` 被 action_head 引用，确保动作头的输入维度与数据变换的输出维度一致。
- **全局参数依赖**：`state_horizon` 和 `action_horizon` 来自数据配置文件（如 `base_48_wan_fine_aug_relative.yaml`）。

## 总结

`dreamzero_cotrain.yaml` 是连接原始数据和模型输入的桥梁。它定义了如何将多种机器人的异构数据（不同的状态/动作维度、不同的视角数、不同的语言指令）统一转换为 DreamZero 模型可处理的标准化格式。通过 UMT5-XXL 分词器处理多语言指令、通过维度填充统一多机器人的状态/动作空间，实现了高效的多任务联合训练。
