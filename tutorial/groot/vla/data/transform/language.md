# 语言变换 (`groot/vla/data/transform/language.py`)

## 文件概述

`language.py` 实现了两种语言模态的变换：`LanguageTransform`（Tokenization 变换）和 `LanguageRemovePrefix`（前缀移除变换）。它们用于将自然语言任务描述转换为模型可处理的格式。

## 关键代码解析

### 1. `LanguageTransform` 类

```python
class LanguageTransform(InvertibleModalityTransform):
    apply_to: list[str]           # 目标键列表
    tokenizer: T_Tokenizer        # HuggingFace tokenizer
```

#### Tokenizer 验证

支持传入字符串路径或已实例化的 tokenizer：

```python
@field_validator("tokenizer")
def validate_tokenizer(cls, v):
    if isinstance(v, str):
        return AutoTokenizer.from_pretrained(v)
    return v
```

#### `apply` 方法

对文本进行 tokenization，返回 PyTorch 张量：

```python
def apply(self, data):
    for key in self.apply_to:
        data[key] = self.tokenizer(
            data[key], return_tensors="pt", padding=True, truncation=True
        ).input_ids
    return data
```

#### `unapply` 方法

将 token IDs 还原为文本：

```python
def unapply(self, data):
    for key in self.apply_to:
        data[key] = self.tokenizer.decode(data[key], skip_special_tokens=True)
    return data
```

### 2. `LanguageRemovePrefix` 类

```python
class LanguageRemovePrefix(ModalityTransform):
    apply_to: list[str]
```

移除语言标注中的前缀（如 `"task: pick up the cup"` -> `"pick up the cup"`），通过 `": "` 分割并取最后一部分：

```python
def apply(self, data):
    for key in self.apply_to:
        value = data[key]
        if isinstance(value[0], np.ndarray):
            # 批量处理: (B, T) 形状
            data[key] = np.array(
                [[lang.split(": ")[-1] for lang in sublist] for sublist in value]
            )
        else:
            # 非批量: (T,) 形状
            data[key] = np.array([lang.split(": ")[-1] for lang in value])
    return data
```

注意此变换区分了批量和非批量两种输入格式。

## 核心类/函数表格

| 名称 | 类型 | 说明 |
|------|------|------|
| `LanguageTransform` | InvertibleModalityTransform | 使用 HuggingFace tokenizer 进行文本编码/解码 |
| `LanguageRemovePrefix` | ModalityTransform | 移除文本前缀（基于 `": "` 分割） |
| `T_Tokenizer` | 类型别名 | `Union[PreTrainedTokenizer, PreTrainedTokenizerFast]` |

## 与其他模块的关系

- **继承 `base.py`** - `LanguageTransform` 继承 `InvertibleModalityTransform`，`LanguageRemovePrefix` 继承 `ModalityTransform`
- **依赖 HuggingFace Transformers** - 使用 `AutoTokenizer` 加载预训练 tokenizer
- **被 `dataset/lerobot.py` 使用** - 在数据加载后对语言标注进行处理

## 总结

语言变换模块提供了文本数据的预处理能力。`LanguageTransform` 通过 HuggingFace tokenizer 将自然语言转换为 token ID 序列，支持正向和逆向转换。`LanguageRemovePrefix` 是一个简单但实用的预处理步骤，用于清理标注数据中的冗余前缀信息。
