# Embodiment 标签枚举 (`groot/vla/data/schema/embodiment_tags.py`)

## 文件概述

`embodiment_tags.py` 定义了 `EmbodimentTag` 枚举类，用于标识 DreamZero 系统中支持的所有机器人构型（embodiment）。每个标签代表一种特定的机器人硬件配置，包括手臂类型、手部类型、是否包含腰部等。

## 关键代码解析

### `EmbodimentTag` 枚举

```python
class EmbodimentTag(Enum):
    REAL_GR1_ARMS_ONLY = "real_gr1_arms_only"
    REAL_GR1_ARMS_WAIST = "real_gr1_arms_waist"
    ...
```

枚举值分为以下几大类：

#### 真实 GR1 机器人

| 标签 | 说明 |
|------|------|
| `REAL_GR1_ARMS_ONLY` | GR1 仅手臂 |
| `REAL_GR1_ARMS_ONLY_ANNOTATED` | GR1 仅手臂（带标注） |
| `REAL_GR1_ARMS_WAIST` | GR1 手臂+腰部 |
| `REAL_GR1_ARMS_WAIST_ANNOTATED` | GR1 手臂+腰部（带标注） |

#### DEXMG 模拟数据

| 标签 | 说明 |
|------|------|
| `DEXMG_GR1_ARMS_ONLY_INSPIRE` | DEXMG GR1 + Inspire 手 |
| `DEXMG_GR1_ARMS_ONLY_FOURIER` | DEXMG GR1 + Fourier 手 |
| `DEXMG_GR1_ARMS_WAIST_FOURIER` | DEXMG GR1 + 腰部 + Fourier 手 |

#### RoboCasa 模拟数据

包含多种配置：单臂、双臂 Panda、GR1 等，搭配不同手型（Inspire、Fourier、平行夹爪）。

#### Open X-Embodiment (OXE) 数据集

| 标签 | 说明 |
|------|------|
| `OXE_DROID` | DROID 数据集 |
| `OXE_FRACTAL` | RT-1 数据集 |
| `OXE_BRIDGE` | Bridge 数据集 |
| `OXE_LANGUAGE_TABLE` | Language Table 数据集 |

#### Unitree G1 机器人

大量 G1 变体，包括：
- 上半身/全身
- 带/不带高度控制
- 带/不带末端执行器控制
- 带/不带导航指令
- 仿真/真实环境

#### 特殊标签

| 标签 | 说明 |
|------|------|
| `DREAM` | DreamZero 专用动作格式 |
| `LAPA` | LAPA 动作格式 |
| `UNKNOWN` | 未知构型 |
| `GR1_UNIFIED` | GR1 统一数据集 |

#### 已弃用标签

以 `DEPRECATED_` 前缀标记的标签表示旧版 G1 构型，主要原因是：
1. 头部未锁定导致相机角度不一致
2. 视频模态键命名不统一（`rs_view` vs `ego_view`）

## 核心类/函数表格

| 名称 | 类型 | 说明 |
|------|------|------|
| `EmbodimentTag` | Enum | 机器人构型标签枚举，包含 50+ 种配置 |

## 与其他模块的关系

- **被 `schema/lerobot.py` 使用** - `DatasetMetadata` 包含 `EmbodimentTag` 字段
- **被 `dataset/registry.py` 使用** - 作为数据集注册表的键
- **被 `dataset/lerobot.py` 使用** - 数据集初始化时指定构型标签
- **被 `conversion/gr1/constants.py` 使用** - 构型映射

## 总结

`EmbodimentTag` 是 DreamZero 多机器人支持的核心抽象。通过枚举化管理所有支持的机器人构型，系统可以根据标签自动加载对应的元数据配置（模态定义、统计信息等），实现了跨机器人平台的统一数据处理管线。标签的命名遵循 `{来源}_{机器人}_{部位}_{手型}` 的规范。
