# `common.py` -- 通用文本处理工具

## 文件概述

本文件提供了语言指令的标准化处理函数，用于统一不同数据集中语言标注的格式。

## 关键代码解析

```python
def formalize_language(language: str) -> str:
    """
    1. Force lowercase
    2. Remove all punctuations
    """
    language = language.lower()
    language = re.sub(r"[^\w\s]", "", language)
    return language
```

`formalize_language` 执行两步操作：
1. **转小写**: 消除大小写差异（如 "Pick up" 和 "pick up" 统一为 "pick up"）
2. **去除标点**: 使用正则表达式移除所有非字母数字和空白字符

## 核心类/函数表格

| 函数名 | 功能描述 |
|--------|---------|
| `formalize_language()` | 文本标准化：转小写 + 去标点 |

## 与其他模块的关系

- **被 `dreamzero_cotrain.py` 调用**: 当 `DreamTransform.formalize_language` 配置为 True 时使用

## 总结

简单但重要的文本预处理工具，通过标准化语言指令格式来减少训练数据中的噪声，提升模型对自然语言指令的泛化能力。
