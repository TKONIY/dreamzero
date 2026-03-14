# DreamZero DROID 全参数微调训练脚本详解

> 对应文件：`scripts/train/droid_training_full_finetune.sh`

## 文件概述

`droid_training_full_finetune.sh` 是 DreamZero 在 DROID 数据集上进行全参数微调（Full Fine-Tuning）的训练脚本。与 LoRA 仅训练少量适配器参数不同，全参数微调会更新模型中的所有可训练参数，通常能获得更好的性能，但需要更多的 GPU 显存和更长的训练时间。

该脚本专为 **8x H100 GPU** 环境设计，使用 DeepSpeed ZeRO-2 + CPU Offload 策略来管理 14B 参数模型的显存需求。

## 关键代码解析

### 1. 与 LoRA 脚本的核心差异

全参数微调脚本与 LoRA 脚本（`droid_training_lora.sh`）共享大部分结构，关键差异如下：

**训练架构：**
```bash
train_architecture=full          # LoRA 版本为 lora
```

**学习率：**
```bash
training_args.learning_rate=1e-5  # LoRA 版本为 1e-4
```

全参数微调使用更小的学习率（低 10 倍），因为需要更新所有参数，过大的学习率可能破坏预训练权重中的知识。

**DeepSpeed 配置：**
```bash
training_args.deepspeed="groot/vla/configs/deepspeed/zero2_offload.json"  # LoRA 版本为 zero2.json
```

使用 `zero2_offload.json` 而非 `zero2.json`，启用 CPU Offload 将部分优化器状态卸载到 CPU 内存，以减少 GPU 显存占用。

**保存策略：**
```bash
save_lora_only=false  # LoRA 版本为 true
```

全参数微调需要保存完整的模型权重，而非仅 LoRA 适配器。

### 2. 完整参数对比

| 参数 | 全参数微调 | LoRA 微调 | 说明 |
|------|-----------|----------|------|
| `train_architecture` | `full` | `lora` | 训练模式 |
| `learning_rate` | `1e-5` | `1e-4` | 全参数微调需要更小学习率 |
| `deepspeed` | `zero2_offload.json` | `zero2.json` | 全参数需要 CPU Offload |
| `save_lora_only` | `false` | `true` | 全参数保存完整权重 |
| `save_strategy` | `no` | `no` | 均为默认不保存中间检查点 |

其他参数（如 `num_frames=33`、`action_horizon=24`、`num_views=3` 等）完全相同。

### 3. DeepSpeed ZeRO-2 + CPU Offload

ZeRO-2 + CPU Offload 的工作原理：

- **ZeRO Stage 2**：将优化器状态和梯度在多个 GPU 之间分片，每个 GPU 只存储全部参数的 1/N
- **CPU Offload**：将优化器状态从 GPU 显存卸载到主机 CPU 内存，进一步降低 GPU 显存需求
- 代价是增加了 GPU-CPU 之间的数据传输开销

这使得 14B 参数的 DreamZero 模型可以在 8x H100（80GB）上进行全参数训练。

### 4. 显存需求估算

对于 14B 参数的模型，全参数微调的显存需求：
- 模型参数（BF16）：~28GB
- 梯度（BF16）：~28GB（ZeRO-2 分片后 ~3.5GB/GPU）
- 优化器状态（FP32）：~56GB（CPU Offload 后不占 GPU 显存）
- 激活值：取决于 `max_chunk_size` 和 `frame_seqlen`

## 核心类/函数表格

| 组件 | 文件路径 | 功能 |
|------|---------|------|
| 训练主入口 | `groot/vla/experiment/experiment.py` | Hydra 驱动的训练主函数 |
| DROID 数据配置 | `groot/vla/configs/data/dreamzero/droid_relative.yaml` | DROID 相对动作数据配置 |
| DeepSpeed 配置 | `groot/vla/configs/deepspeed/zero2_offload.json` | ZeRO-2 + CPU Offload 配置 |
| 模型配置 | `groot/vla/configs/model/dreamzero/vla.yaml` | DreamZero VLA 模型定义 |

## 与其他模块的关系

- **LoRA 对比**：与 `droid_training_lora.sh` 形成对照实验，可用 `scripts/compare_loss.py` 对比两者的训练损失曲线
- **数据依赖**：与 LoRA 脚本相同，需要 LeRobot 格式的 DROID 数据集
- **检查点输出**：默认输出到 `./checkpoints/dreamzero_droid_full_finetune`，保存完整模型权重
- **硬件要求**：至少需要 8x H100 或同等显存级别的 GPU

## 总结

`droid_training_full_finetune.sh` 通过全参数微调在 DROID 数据集上训练 DreamZero 模型。相比 LoRA 微调，全参数微调更新所有模型参数，潜在性能更好，但需要显著更多的计算资源。脚本通过 DeepSpeed ZeRO-2 + CPU Offload 策略优化显存管理，使得 14B 模型可以在 8x H100 上训练。主要使用场景是当 LoRA 微调的性能无法满足需求，且有充足 GPU 资源时的深度微调。
