# conf.yaml -- DreamZero VLA 主配置文件

## 文件概述

`conf.yaml` 是 DreamZero VLA（Vision-Language-Action）训练系统的**顶层主配置文件**。它基于 [Hydra](https://hydra.cc/) 配置框架，通过 `defaults` 列表组合模型配置和数据配置，并定义了完整的训练超参数、优化器设置、检查点策略、性能分析选项等。所有其他子配置文件（模型、数据、DeepSpeed 等）都通过此文件汇聚在一起，构成最终的训练配置。

## 关键配置项解析

### 1. Hydra Defaults 组合

```yaml
defaults:
  - _self_
  - model: dreamzero/vla
  - data: dreamzero/droid_horizon_relative
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled
```

- `_self_`：表示当前文件的配置优先级最低，后续加载的子配置会覆盖此文件中的同名字段。
- `model: dreamzero/vla`：引入模型配置，指向 `model/dreamzero/vla.yaml`。
- `data: dreamzero/droid_horizon_relative`：引入数据配置（默认为 DROID 数据集）。
- 禁用 Hydra 自身的日志输出，保持控制台整洁。

### 2. Trainer 配置

```yaml
trainer:
  _target_: groot.vla.experiment.VLATrainer
```

Trainer 使用 Hydra 的 `_target_` 实例化机制，指向 `groot.vla.experiment.VLATrainer` 类。关键子项包括：
- `benchmark_time`：是否在训练中进行时间基准测试。
- `enable_profiling` / `profiling_steps`：传统逐步性能分析。
- `enable_prof_callback`：基于窗口的性能分析回调（ProfCallback），可配置启动步数、预热步数、活跃步数等。

### 3. 优化器与学习率调度

使用 AdamW 优化器，配合 cosine 学习率调度器，包含预热阶段。

### 4. 检查点保存策略

通过 `save_strategy: steps` 和 `save_steps: 500` 每 500 步保存一次，最多保留 8 个检查点。支持上传检查点到远程存储（通过 `upload_checkpoints` 等选项）。

### 5. training_args 转发

`training_args` 块使用 `_target_: transformers.TrainingArguments` 将顶层参数通过 Hydra 的 `${}` 插值语法转发给 HuggingFace `TrainingArguments`，避免重复定义。

## 配置参数表格

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `wandb_project` | `???`（必填） | Weights & Biases 项目名 |
| `output_dir` | `???`（必填） | 输出目录 |
| `bf16` | `false` | 是否使用 BF16 混合精度训练 |
| `tf32` | `false` | 是否启用 TF32 加速 |
| `global_batch_size` | `null` | 全局批次大小（可选） |
| `per_device_train_batch_size` | `256` | 每设备训练批次大小 |
| `per_device_eval_batch_size` | `64` | 每设备评估批次大小 |
| `gradient_accumulation_steps` | `1` | 梯度累积步数 |
| `dataloader_num_workers` | `10` | 数据加载器工作线程数 |
| `dataloader_pin_memory` | `true` | 是否使用锁页内存 |
| `dataloader_persistent_workers` | `true` | 是否保持工作线程存活 |
| `optim` | `adamw_torch` | 优化器类型 |
| `learning_rate` | `1e-4` | 学习率 |
| `adam_beta1` | `0.95` | Adam 优化器 beta1 |
| `adam_beta2` | `0.999` | Adam 优化器 beta2 |
| `adam_epsilon` | `1e-8` | Adam 优化器 epsilon |
| `weight_decay` | `1e-6` | 权重衰减 |
| `lr_scheduler_type` | `cosine` | 学习率调度类型 |
| `warmup_ratio` | `0.05` | 预热比例 |
| `logging_steps` | `10.0` | 日志记录间隔步数 |
| `num_train_epochs` | `1000` | 最大训练轮数 |
| `max_steps` | `-1` | 最大训练步数（-1 表示不限制） |
| `save_strategy` | `steps` | 保存策略 |
| `save_steps` | `500` | 保存间隔步数 |
| `save_total_limit` | `8` | 最多保存检查点数 |
| `report_to` | `wandb` | 日志上报目标 |
| `seed` | `42` | 随机种子 |
| `gradient_checkpointing` | `false` | 是否启用梯度检查点 |
| `ddp_find_unused_parameters` | `false` | DDP 是否查找未使用参数 |
| `ddp_bucket_cap_mb` | `100` | DDP 通信桶大小（MB） |
| `pretrained_model_path` | `null` | 预训练模型路径 |
| `only_tune_projectors` | `false` | 是否仅微调投影层 |
| `save_llm` | `false` | 是否保存 LLM 部分 |
| `save_lora_only` | `false` | 是否仅保存 LoRA 权重 |

## 与其他配置/模块的关系

- **模型配置**：通过 `defaults` 引入 `model/dreamzero/vla.yaml`，该文件进一步引入 backbone、action_head、transform 子配置。
- **数据配置**：通过 `defaults` 引入 `data/dreamzero/` 下的数据集配置文件。
- **DeepSpeed 配置**：通过 `training_args.deepspeed` 字段指定 DeepSpeed JSON 配置文件路径。
- **HuggingFace TrainingArguments**：`training_args` 块将训练参数转发给 `transformers.TrainingArguments`。
- **Hydra**：底部 `hydra` 块禁用了 Hydra 的输出子目录和运行目录切换，保持工作目录不变。

## 总结

`conf.yaml` 是 DreamZero VLA 训练流水线的中枢配置文件。它通过 Hydra 的组合机制将模型定义、数据管道、训练超参数、性能分析选项和检查点策略统一管理。用户启动训练时只需在命令行覆盖 `wandb_project`、`output_dir` 等必填项，以及按需切换数据集配置即可。所有参数均支持通过 Hydra CLI 的 `key=value` 语法进行覆盖。
