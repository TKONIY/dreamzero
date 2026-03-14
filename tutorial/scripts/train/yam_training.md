# DreamZero YAM 训练脚本详解

> 对应文件：`scripts/train/yam_training.sh`

## 文件概述

`yam_training.sh` 是 DreamZero 在 YAM 数据集上进行 LoRA 微调的训练脚本。YAM 是一种双臂机器人平台，状态和动作空间均为 14 维，包含 3 个相机视角（top、left、right）。

该脚本与 `agibot_training.sh` 结构高度相似，同样采用预训练检查点 + LoRA 微调的迁移学习策略，但训练规模显著更大（100,000 步，每 GPU 批大小 4）。

## 关键代码解析

### 1. 数据集配置

```bash
YAM_DATA_ROOT=${YAM_DATA_ROOT:-"./data/yam_lerobot"}
```

YAM 数据集需要满足以下要求：
- `meta/embodiment.json` 中包含 `"embodiment_tag": "yam"`
- **状态**：14 维（`left_joint_pos` + `left_gripper_pos` + `right_joint_pos` + `right_gripper_pos`）
- **动作**：14 维（与状态相同的关键字段）
- **视频**：3 个视角（top、left、right）
- **标注**：`annotation.human.task_description` 包含人工语言指令

### 2. 大规模训练配置

与其他训练脚本的关键差异：

| 参数 | YAM | AGIBot | DROID LoRA |
|------|-----|--------|-----------|
| `max_steps` | **100,000** | 5,000 | 100 |
| `per_device_train_batch_size` | **4** | 1 | 1 |
| `save_steps` | **10,000** | 2,500 | 1,000 |
| `learning_rate` | `1e-5` | `1e-5` | `1e-4` |
| `data` | `dreamzero/yam_relative` | `dreamzero/agibot_relative` | `dreamzero/droid_relative` |

YAM 脚本使用 4 倍的批大小和 100 倍的训练步数（相比 DROID），表明 YAM 是一个更正式的大规模训练场景。更大的批大小得益于 YAM 数据集可能包含更小的状态/动作维度（14 vs DROID 的 28/14）。

### 3. 预训练模型加载

```bash
pretrained_model_path=./checkpoints/DreamZero-AgiBot \
++action_head_cfg.config.skip_component_loading=true \
++action_head_cfg.config.defer_lora_injection=true
```

与 AGIBot 脚本相同，YAM 也加载 DreamZero-AgiBot 预训练检查点。这说明 DreamZero-AgiBot 作为一个通用的预训练基础模型，可以迁移到不同的机器人平台：

```
DreamZero-AgiBot（预训练）
    ├──→ AGIBot 微调（agibot_training.sh）
    └──→ YAM 微调（yam_training.sh）
```

通过 `skip_component_loading` 和 `defer_lora_injection`，即使 YAM 的动作空间（14 维）与 AGIBot（22 维）不同，也能正确加载预训练的视觉和语言特征。

### 4. 输出目录命名

```bash
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_yam_lora_dz_pretrained_100k_folding"}
```

默认输出目录名包含了训练的关键信息：
- `yam`：数据集
- `lora`：训练策略
- `dz_pretrained`：使用 DreamZero 预训练权重
- `100k`：100,000 步训练
- `folding`：暗示特定的任务类型（折叠）

### 5. 数据配置

```bash
data=dreamzero/yam_relative \
yam_data_root=$YAM_DATA_ROOT \
```

使用 `yam_relative` 数据配置，在 Hydra 配置系统中对应 `groot/vla/configs/data/dreamzero/yam_relative.yaml`，定义了 YAM 数据集的加载方式和字段映射。

## 核心类/函数表格

| 组件 | 文件路径 | 功能 |
|------|---------|------|
| 训练主入口 | `groot/vla/experiment/experiment.py` | Hydra 驱动的训练主函数 |
| YAM 数据配置 | `groot/vla/configs/data/dreamzero/yam_relative.yaml` | YAM 相对动作数据配置 |
| 预训练检查点 | `checkpoints/DreamZero-AgiBot/` | 预训练模型权重 |
| DeepSpeed 配置 | `groot/vla/configs/deepspeed/zero2.json` | ZeRO-2 优化器配置 |
| 开环评估 | `scripts/open_loop_yam.py` | YAM 模型的离线评估工具 |

## 与其他模块的关系

- **预训练模型**：与 `agibot_training.sh` 共享同一个 DreamZero-AgiBot 预训练检查点
- **开环评估**：训练完成后可使用 `scripts/open_loop_yam.py` 进行离线评估，该脚本通过 `GrootSimPolicy` 加载训练好的 YAM 模型
- **数据准备**：YAM 数据需要预先转换为包含正确 `embodiment.json` 的 LeRobot v2 格式
- **推理策略**：`groot/vla/model/n1_5/sim_policy.py` 中的 `GrootSimPolicy` 使用 `EmbodimentTag.YAM` 加载 YAM 模型

## 总结

`yam_training.sh` 是 DreamZero 项目中训练规模最大的脚本（100K 步、批大小 4）。它在 YAM 双臂机器人数据集上进行 LoRA 微调，通过加载 DreamZero-AgiBot 预训练检查点实现跨平台迁移学习。脚本的自动 GPU 检测和迁移学习机制使其在不同硬件和数据环境下都能灵活使用。训练产生的模型可通过 `open_loop_yam.py` 进行开环动作预测评估。
