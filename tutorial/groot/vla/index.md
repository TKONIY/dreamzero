# DreamZero VLA 模块概览

## 概述

`groot/vla/` 是 DreamZero 的核心 VLA（Vision-Language-Action）实现目录，包含从模型架构到训练流程的完整代码。

## 子模块索引

| 子模块 | 路径 | 说明 |
|--------|------|------|
| [common](common/index.md) | `groot/vla/common/` | 通用工具库：IO、数据结构、张量操作、图像处理等 |
| [configs](configs/index.md) | `groot/vla/configs/` | Hydra 配置文件：模型、数据、DeepSpeed 等 |
| [data](data/index.md) | `groot/vla/data/` | 数据管线：数据集加载、schema 定义、数据变换、格式转换 |
| [experiment](experiment/index.md) | `groot/vla/experiment/` | 实验与训练框架：基类、VLA 训练器、工具函数 |
| [model](model/index.md) | `groot/vla/model/` | 模型实现：DreamZero VLA、backbone、action head、DiT 模块 |
| [utils](utils/index.md) | `groot/vla/utils/` | 辅助工具：计时器、动作参数覆盖等 |

## 架构关系

```
configs/ (配置定义)
  │
  v
experiment/ (训练入口) ──> model/ (模型实现)
  │                          │
  v                          v
data/ (数据管线)          common/ (通用工具)
                            │
                            v
                          utils/ (辅助工具)
```
