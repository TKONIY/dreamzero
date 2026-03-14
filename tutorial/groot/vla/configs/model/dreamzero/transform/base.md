# base.yaml -- 机器人体型映射基础配置

## 文件概述

`base.yaml` 是 DreamZero transform 模块的**基础配置文件**，定义了**机器人体型标签到投影器索引的映射表**（`embodiment_tag_to_projector_index`）。DreamZero 支持多种不同的机器人平台和形态（embodiment），每种机器人有不同的状态和动作空间维度。此映射表用于在模型中为每种机器人分配独立的投影器（projector），实现多机器人联合训练。

## 关键配置项解析

### embodiment_tag_to_projector_index

该映射表将每种机器人体型的字符串标签映射到一个整数索引。模型内部使用这个索引来选择对应的状态/动作投影器。

支持的机器人体型包括：

- **GR-1 系列**（人形机器人）：`real_gr1_arms_only`、`real_gr1_arms_waist`、`gr1_unified` 等
- **仿真机器人**：`robocasa_single_arm`、`robocasa_bimanual_panda_*`、`language_table_sim` 等
- **OXE 数据集机器人**：`oxe_droid`(17)、`oxe_fractal`(18)、`oxe_bridge`(20) 等
- **AGIBot**：`agibot`(26)
- **YAM**：`yam`(32)
- **其他**：`lapa`(27)、`dream`(31)、`xdof`(22) 等

注意部分索引存在复用（如索引 0 同时用于 `real_gr1_arms_only` 和 `gr1_isaac`），这意味着这些机器人共享相同的投影器。

## 配置参数表格

| 机器人体型标签 | 投影器索引 | 说明 |
|---------------|-----------|------|
| `real_gr1_arms_only` | 0 | GR-1 仅手臂 |
| `real_gr1_arms_only_annotated` | 1 | GR-1 仅手臂（带标注） |
| `real_gr1_arms_waist` | 2 | GR-1 手臂+腰部 |
| `real_gr1_arms_waist_annotated` | 3 | GR-1 手臂+腰部（带标注） |
| `dexmg_gr1_arms_only_inspire` | 4 | DexMG GR-1 Inspire 手 |
| `dexmg_gr1_arms_only_fourier` | 5 | DexMG GR-1 Fourier 手 |
| `dexmg_gr1_arms_waist_fourier` | 6 | DexMG GR-1 腰部 Fourier |
| `robocasa_single_arm` | 7 | RoboCasa 单臂 |
| `onex_eve_gripper` | 8 | OneX Eve 夹爪 |
| `oxe_droid` | 17 | OXE DROID 数据集 |
| `oxe_fractal` | 18 | OXE Fractal 数据集 |
| `oxe_bridge` | 20 | OXE Bridge 数据集 |
| `real_panda_single_arm` | 21 | Panda 单臂 |
| `gr1_unified` | 24 | GR-1 统一配置 |
| `agibot` | 26 | AGIBot 机器人 |
| `lapa` | 27 | LAPA 机器人 |
| `yam` | 32 | YAM 机器人 |
| `dream` | 31 | Dream 机器人 |

## 与其他配置/模块的关系

- **被引用**：由 `transform/dreamzero_cotrain.yaml` 通过 `defaults` 中的 `- base` 引入。
- **在数据变换中使用**：`dreamzero_cotrain.yaml` 中的 `DreamTransform` 和 `DefaultDataCollator` 都使用此映射表来确定每个样本对应的投影器索引。
- **在 action_head 中使用**：`action_loss_embodiment_ids: [26, 17, 32]` 引用了此表中的 agibot(26)、oxe_droid(17)、yam(32) 的索引。

## 总结

`base.yaml` 是 DreamZero 多机器人联合训练能力的基础。通过维护一个体型标签到投影器索引的映射表，系统可以在单个模型中同时支持数十种不同的机器人平台。每种机器人使用独立的投影器来处理各自不同维度的状态和动作空间，实现了灵活的多体型扩展。添加新机器人类型时，只需在此表中分配一个新的索引即可。
