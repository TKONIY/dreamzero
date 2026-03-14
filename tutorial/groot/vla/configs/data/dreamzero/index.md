# 数据集配置索引

## 概述

`data/dreamzero/` 目录包含 DreamZero 训练所需的数据集配置文件。所有配置都继承自一个基础文件，定义了统一的数据管道（视频帧数、分辨率、增强策略、模态结构），各具体配置只需指定数据路径和机器人特定的相对动作键。

## 文件索引

### 基础配置
- [base_48_wan_fine_aug_relative.yaml](base_48_wan_fine_aug_relative.md) -- 基础数据管道配置（49 帧视频、48 步动作、数据增强、多数据集模态定义）

### 单数据集训练配置
- [droid_relative.yaml](droid_relative.md) -- DROID 数据集（单臂操作，7 维关节+1 维夹爪）
- [agibot_relative.yaml](agibot_relative.md) -- AGIBot 数据集（人形双臂，6 组关节，30+ 维状态/动作）
- [yam_relative.yaml](yam_relative.md) -- YAM 数据集（双臂平台，4 组关节，14 维状态/动作）

## 数据集对比

| 数据集 | 体型标签 | 投影器索引 | 视角数 | 状态维度 | 动作维度 | 相对动作键数 |
|--------|---------|-----------|-------|---------|---------|------------|
| DROID | `oxe_droid` | 17 | 3 | ~8 | ~8 | 1 |
| AGIBot | `agibot` | 26 | 3 | ~32 | ~30 | 6 |
| YAM | `yam` | 32 | 3 | ~14 | ~14 | 4 |

## 配置继承关系

```
base_48_wan_fine_aug_relative.yaml
  ├─ droid_relative.yaml     (覆盖: 数据路径, relative_action_keys)
  ├─ agibot_relative.yaml    (覆盖: 数据路径, relative_action_keys)
  └─ yam_relative.yaml       (覆盖: 数据路径, relative_action_keys)
```

## 添加新数据集

要添加新的机器人数据集，需要：
1. 在 `base_48_wan_fine_aug_relative.yaml` 中添加模态配置和变换链。
2. 在 `transform/base.yaml` 中添加体型标签映射。
3. 创建新的 `*_relative.yaml` 继承基础配置，指定数据路径和相对动作键。
