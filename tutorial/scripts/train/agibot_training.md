# DreamZero AGIBot 训练脚本详解

> 对应文件：`scripts/train/agibot_training.sh`

## 文件概述

`agibot_training.sh` 是 DreamZero 在 AGIBot 数据集上进行 LoRA 微调的训练脚本。AGIBot 是一种双臂人形机器人平台，状态空间为 32 维（后经处理为 20 维），动作空间为 22 维，包含 3 个相机视角（top_head、hand_left、hand_right）。

与 DROID 训练脚本相比，该脚本有两个重要特点：
1. 自动检测可用 GPU 数量，适应不同硬件环境
2. 加载 DreamZero-AgiBot 预训练检查点进行迁移学习

## 关键代码解析

### 1. 自动检测 GPU 数量

```bash
if [ -z "${NUM_GPUS}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}
```

与 DROID 脚本的固定 `NUM_GPUS=${NUM_GPUS:-8}` 不同，AGIBot 脚本先尝试通过 `nvidia-smi` 检测实际可用的 GPU 数量。这使得脚本可以在 4-GPU 或 8-GPU 机器上无需修改直接运行。

### 2. 数据集配置

```bash
AGIBOT_DATA_ROOT=${AGIBOT_DATA_ROOT:-"./data/agibot_lerobot"}
```

AGIBot 数据集需要以 LeRobot 格式存储，包含以下特征：
- **状态**：20 维（左臂关节 7 + 右臂关节 7 + 左夹爪 1 + 右夹爪 1 + 头部 2 + 腰部 2）
- **动作**：22 维（状态维度 + 机器人速度 2）
- **视频**：3 个视角（top_head、hand_left、hand_right）

### 3. 预训练模型加载与迁移学习

```bash
pretrained_model_path=./checkpoints/DreamZero-AgiBot \
++action_head_cfg.config.skip_component_loading=true \
++action_head_cfg.config.defer_lora_injection=true
```

这是 AGIBot 脚本与 DROID 脚本的核心差异。这三个参数实现了迁移学习流程：

- **`pretrained_model_path`**：指定预训练的 DreamZero-AgiBot 检查点路径。该检查点通过 `git clone https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot` 获取
- **`skip_component_loading=true`**：跳过动作头组件的权重加载，因为预训练检查点的动作头可能与当前任务配置不匹配
- **`defer_lora_injection=true`**：延迟 LoRA 适配器的注入时机，确保先加载基础模型权重再注入 LoRA 层

`++` 前缀是 Hydra 的覆盖语法，表示在默认配置之外追加新的配置键。

### 4. 训练超参数差异

与 DROID LoRA 脚本的参数对比：

| 参数 | AGIBot | DROID LoRA | 说明 |
|------|--------|-----------|------|
| `data` | `dreamzero/agibot_relative` | `dreamzero/droid_relative` | 不同数据集配置 |
| `learning_rate` | `1e-5` | `1e-4` | AGIBot 使用更小学习率（迁移学习） |
| `max_steps` | `5000` | `100` | AGIBot 训练更多步 |
| `save_steps` | `2500` | `1000` | 每 2500 步保存检查点 |
| `save_strategy` | `steps` | `no` | AGIBot 启用定期保存 |
| `report_to` | `wandb` | `none` | AGIBot 启用 WandB 日志 |
| `per_device_train_batch_size` | `1` | `1` | 相同 |

### 5. WandB 集成

```bash
report_to=wandb \
wandb_project=dreamzero \
```

AGIBot 脚本默认启用 Weights & Biases 实验跟踪，方便监控训练过程中的损失曲线和其他指标。首次运行需要配置 `wandb login`。

## 核心类/函数表格

| 组件 | 文件路径 | 功能 |
|------|---------|------|
| 训练主入口 | `groot/vla/experiment/experiment.py` | Hydra 驱动的训练主函数 |
| AGIBot 数据配置 | `groot/vla/configs/data/dreamzero/agibot_relative.yaml` | AGIBot 相对动作数据配置 |
| 预训练检查点 | `checkpoints/DreamZero-AgiBot/` | HuggingFace 上的预训练模型 |
| DeepSpeed 配置 | `groot/vla/configs/deepspeed/zero2.json` | ZeRO-2 优化器配置 |
| 数据转换 | `scripts/data/convert_agibot.py` | AGIBot 原始数据转换工具 |

## 与其他模块的关系

- **数据准备**：AGIBot 原始数据需要通过 `scripts/data/convert_agibot.py` 转换为 LeRobot 格式。详见 `docs/DATASET_TO_GEAR_AND_TRAIN.md`
- **预训练依赖**：需要先下载 `GEAR-Dreams/DreamZero-AgiBot` 预训练检查点
- **YAM 训练延伸**：`yam_training.sh` 使用相同的预训练模型加载机制，可视为 AGIBot 训练的扩展
- **模型评估**：训练产生的检查点可通过类似 `scripts/open_loop_yam.py` 的评估脚本进行验证

## 总结

`agibot_training.sh` 在 AGIBot 双臂人形机器人数据集上进行 DreamZero 的 LoRA 微调训练。脚本的核心特色是：自动 GPU 检测适配不同硬件、加载预训练检查点实现迁移学习、延迟 LoRA 注入确保权重正确加载。较小的学习率（`1e-5`）和较多的训练步数（5000）反映了迁移学习场景下的调参策略。
