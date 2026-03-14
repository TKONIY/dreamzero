# `dreamzero_cotrain.py` -- DreamZero 联合训练数据变换

## 文件概述

本文件实现了 DreamZero 的数据预处理管线，负责将来自不同机器人形态（embodiment）的原始数据统一转换为模型可消费的格式。核心功能包括：

1. **HuggingfaceTokenizer**: 文本分词器封装
2. **DefaultDataCollator**: 数据批次整理器，处理文本分词和张量堆叠
3. **DreamTransform**: 核心数据变换类，处理多视角视频拼接、语言标注、状态/动作填充、以及多种训练模式（LAPA、DREAM、COTRAIN）

## 关键代码解析

### HuggingfaceTokenizer -- 分词器封装

```python
class HuggingfaceTokenizer:
    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)

    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop('return_mask', False)
        ids = self.tokenizer(sequence, return_tensors='pt', padding='max_length', ...)
        if return_mask:
            return ids.input_ids, ids.attention_mask
        return ids.input_ids
```

封装了 HuggingFace 的分词器，支持最大长度填充、空格清理和 attention mask 返回。

### collate 函数 -- 批次整理

```python
def collate(features, tokenizer, num_views=3, embodiment_tag_mapping=None):
    for key in keys:
        if key == "text":
            # 根据 embodiment_id 为文本添加视角描述前缀
            if elem["embodiment_id"] == EmbodimentTag.AGIBOT:
                processed_item = "A multi-view video shows that a robot " + item + " The video is split into four views..."
            elif elem["embodiment_id"] == EmbodimentTag.OXE_DROID:
                processed_item = "A multi-view video shows that a robot " + item + " The video is split into three views..."
```

collate 函数的关键设计是为不同机器人形态的语言指令添加视角描述前缀。例如，AGIBOT 的指令会被添加四视角描述，DROID 的指令会被添加三视角描述。这使得文本编码器能够理解不同的视频布局。

### DreamTransform -- 核心数据变换

```python
class DreamTransform(InvertibleModalityTransform):
    def _prepare_video(self, data):
        images = rearrange(data["video"], "t v h w c -> v t c h w")
        if images.shape[0] > 1:
            # 多视角拼接为 2x2 网格
            concat_images = np.zeros((1, t, c, 2*h, 2*w), dtype=images.dtype)
            concat_images[0, :, :, :h, :w] = images[0]   # 左上: 头部相机
            concat_images[0, :, :, h:, :w] = images[1]    # 左下: 左手
            concat_images[0, :, :, :h, w:] = images[2]    # 右上: 右手
            # 右下: 黑色（已初始化为零）
```

多视角视频通过拼接为 2x2 网格统一为单视频输入，这是 DreamZero 处理多视角数据的关键设计。

```python
    def _prepare_action(self, data):
        actions = data["action"]
        # 填充到 max_action_dim
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)))
        # 创建 mask 标识有效维度
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True
        return actions, actions_mask, n_action_tokens
```

动作维度通过零填充统一到 `max_action_dim`，配合 mask 标识有效维度，使得不同关节数的机器人可以共用同一套模型。

### 训练模式支持

```python
    def _prepare_language(self, data):
        if "<LAPA>" in raw_language:
            is_lapa_instance = True   # LAPA 模式
        if "<DREAM>" in raw_language:
            is_dream_instance = True  # DREAM 模式（互联网视频训练）
        if "<COTRAIN>" in raw_language:
            is_cotrain_instance = True  # 联合训练模式
```

通过语言指令中的特殊标记区分不同训练模式，灵活支持多种数据源的混合训练。

## 核心类/函数表格

| 类/函数名 | 功能描述 |
|-----------|---------|
| `HuggingfaceTokenizer` | 分词器封装，支持定长填充 |
| `collate()` | 批次整理，为不同形态添加视角描述 |
| `DefaultDataCollator` | 标准数据整理器 |
| `DreamTransform` | 核心数据变换类 |
| `_prepare_video()` | 多视角视频拼接为 2x2 网格 |
| `_prepare_language()` | 语言预处理，支持多训练模式 |
| `_prepare_state()` | 状态填充到统一维度 |
| `_prepare_action()` | 动作填充到统一维度 |
| `apply_single()` | 处理单个数据样本 |
| `apply_batch()` | 批量处理数据 |

## 与其他模块的关系

- **继承 `InvertibleModalityTransform`**: 来自 `groot.vla.data.transform.base`
- **使用 `common.py`**: 调用 `formalize_language` 进行文本规范化
- **使用 `groot.vla.data.schema`**: 使用 `EmbodimentTag` 区分机器人形态
- **为 `wan_flow_matching_action_tf.py` 提供输入**: 变换后的数据直接送入动作头训练

## 总结

`dreamzero_cotrain.py` 是 DreamZero 数据管线的核心，它解决了多机器人形态、多视角、多训练模式的数据统一化问题。通过视频网格拼接、动作/状态维度填充、以及基于形态的文本前缀，不同来源的数据被转换为统一格式，使得单一模型可以跨形态学习通用的视觉-动作策略。
