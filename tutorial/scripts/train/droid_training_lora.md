# DreamZero DROID LoRA 训练脚本详解

> 对应文件：`scripts/train/droid_training_lora.sh`

## 文件概述

`droid_training_lora.sh` 是 DreamZero 在 DROID 数据集上使用 LoRA（Low-Rank Adaptation）策略进行微调的训练脚本。LoRA 是一种参数高效微调方法，仅训练少量低秩矩阵参数，显著降低显存需求和训练成本，同时保持接近全参数微调的性能。

该脚本是 DreamZero 项目中最基础、最常用的训练入口，适合：
- 首次运行 DreamZero 训练验证环境配置
- 在有限 GPU 资源下进行模型微调
- 快速迭代实验

## 关键代码解析

### 1. 环境与配置初始化

```bash
export HYDRA_FULL_ERROR=1
```

开启 Hydra 的完整错误信息输出，便于调试配置问题。

### 2. 用户可配置变量

```bash
DROID_DATA_ROOT=${DROID_DATA_ROOT:-"./data/droid_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_droid_lora"}
NUM_GPUS=${NUM_GPUS:-8}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
```

每个变量都支持通过环境变量覆盖，默认值适用于标准目录布局。`DROID_DATA_ROOT` 指向已转换为 LeRobot v2 格式的 DROID 数据集。

### 3. 自动下载模型权重

脚本会自动检测 Wan2.1 模型权重和 umt5-xxl 分词器是否存在，缺失时通过 `huggingface-cli download` 自动下载。下载的权重包含以下组件：

- `models_t5_umt5-xxl-enc-bf16.pth`：文本编码器
- `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`：图像编码器
- `Wan2.1_VAE.pth`：VAE 编解码器
- DiT 模型权重（主目录下）

### 4. 核心训练参数详解

```bash
torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
```

**数据相关参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `data=dreamzero/droid_relative` | Hydra 配置组 | DROID 相对动作空间数据配置 |
| `droid_data_root=$DROID_DATA_ROOT` | 数据路径 | LeRobot 格式的 DROID 数据集根目录 |
| `num_frames=33` | 帧数 | 每个训练样本包含 33 帧视频 |
| `num_views=3` | 视角数 | 3 个相机视角 |
| `action_horizon=24` | 动作窗口 | 预测未来 24 步动作 |

**模型架构参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `model=dreamzero/vla` | 模型配置 | DreamZero Vision-Language-Action 模型 |
| `model/dreamzero/action_head=wan_flow_matching_action_tf` | 动作头 | 基于 Wan 的 Flow Matching 动作预测头 |
| `model/dreamzero/transform=dreamzero_cotrain` | 变换 | DreamZero 联合训练数据变换 |
| `train_architecture=lora` | 训练架构 | LoRA 微调模式 |

**训练超参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `training_args.learning_rate=1e-4` | 学习率 | LoRA 通常使用较大学习率 |
| `training_args.warmup_ratio=0.05` | 预热比例 | 总步数的 5% 用于学习率预热 |
| `per_device_train_batch_size=1` | 批大小 | 视频模型显存密集，每 GPU 仅 1 个样本 |
| `max_steps=100` | 最大步数 | 默认 100 步用于快速验证 |
| `weight_decay=1e-5` | 权重衰减 | L2 正则化系数 |
| `seed=42` | 随机种子 | 确保实验可复现 |

**序列化参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_frame_per_block=2` | 块内帧数 | 每个处理块包含 2 帧 |
| `num_action_per_block=24` | 块内动作数 | 每个块预测 24 个动作 |
| `num_state_per_block=1` | 块内状态数 | 每个块包含 1 个状态 |
| `frame_seqlen=880` | 帧序列长度 | 单帧展平后的 token 序列长度 |
| `max_chunk_size=4` | 分块大小 | 最大处理分块 |

**保存与精度参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `save_lora_only=true` | 仅保存 LoRA | 只保存适配器权重，节省磁盘空间 |
| `save_strategy=no` | 保存策略 | 训练中不保存中间检查点 |
| `bf16=true` / `tf32=true` | 混合精度 | 使用 BF16 和 TF32 加速计算 |

## 核心类/函数表格

| 组件 | 文件路径 | 功能 |
|------|---------|------|
| 训练主入口 | `groot/vla/experiment/experiment.py` | 解析 Hydra 配置，初始化模型和训练器 |
| DROID 数据配置 | `groot/vla/configs/data/dreamzero/droid_relative.yaml` | 定义数据加载参数 |
| DeepSpeed 配置 | `groot/vla/configs/deepspeed/zero2.json` | ZeRO-2 内存优化配置 |
| VLA 模型 | `groot/vla/configs/model/dreamzero/vla.yaml` | 模型结构定义 |
| 动作头 | `groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf.yaml` | Flow Matching 动作预测头配置 |

## 与其他模块的关系

- **数据依赖**：需要通过 `scripts/data/convert_droid.py` 预处理的 DROID 数据，或从 HuggingFace 下载 `GEAR-Dreams/DreamZero-DROID-Data`
- **全参数对比**：`droid_training_full_finetune.sh` 使用 `train_architecture=full`，学习率更小（`1e-5`），并启用 CPU Offload
- **下游评估**：训练产生的检查点可用于 `scripts/open_loop_yam.py` 等评估脚本
- **损失分析**：`scripts/compare_loss.py` 可对比 LoRA 和全参数微调的训练曲线

## 总结

`droid_training_lora.sh` 通过 LoRA 微调策略在 DROID 数据集上训练 DreamZero 模型。脚本自动处理权重下载和环境验证，用户只需配置数据路径即可启动训练。LoRA 方式仅更新少量参数，适合 GPU 资源有限的场景。默认 100 步适合环境验证，实际训练应调大步数并启用 `save_strategy=steps`。
