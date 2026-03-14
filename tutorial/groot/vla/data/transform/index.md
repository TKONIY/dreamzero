# 数据变换子模块 (`groot/vla/data/transform`)

## 文件概述

`transform` 子模块提供了可组合的模态数据变换管线，用于对数据集输出的原始数据进行预处理、增强和标准化。变换管线支持正向变换（训练时数据预处理）和逆向变换（推理时动作还原）。

## 文件列表

| 文件 | 说明 | 教程链接 |
|------|------|----------|
| `base.py` | 变换基类定义，包含组合变换和恒等变换 | [base.md](base.md) |
| `concat.py` | 拼接变换，将多个模态键合并为统一张量 | [concat.md](concat.md) |
| `language.py` | 语言变换，包含 tokenization 和前缀移除 | [language.md](language.md) |
| `state_action.py` | 状态/动作变换，包含归一化、旋转转换、扰动等 | [state_action.md](state_action.md) |
| `video.py` | 视频变换，包含裁剪、缩放、颜色增强等 | [video.md](video.md) |

## `__init__.py`

导出所有变换类，包括基类 (`ModalityTransform`, `InvertibleModalityTransform`, `ComposedModalityTransform`) 以及各具体变换实现。

## 变换管线典型流程

```
原始数据 -> VideoToTensor -> VideoResize -> VideoCrop
         -> StateActionToTensor -> StateActionTransform
         -> LanguageTransform
         -> ConcatTransform -> 模型输入
```

所有变换通过 `ComposedModalityTransform` 串联执行。
