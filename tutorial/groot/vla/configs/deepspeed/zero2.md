# zero2.json -- DeepSpeed ZeRO Stage 2 配置

## 文件概述

`zero2.json` 定义了 DeepSpeed ZeRO（Zero Redundancy Optimizer）Stage 2 的分布式训练配置。ZeRO-2 通过在多个 GPU 之间分片优化器状态和梯度来减少显存占用，同时保持每个 GPU 上完整的模型参数副本。这是 DreamZero 默认推荐的分布式训练策略。

## 关键配置项解析

### 1. 自动批次配置

```json
"train_batch_size": "auto",
"train_micro_batch_size_per_gpu": "auto",
"gradient_accumulation_steps": "auto",
"gradient_clipping": "auto"
```

所有批次相关参数设为 `"auto"`，由 HuggingFace Trainer 根据 `conf.yaml` 中的训练参数自动计算。

### 2. 混合精度配置

```json
"fp16": {
  "enabled": "auto",
  "loss_scale": 0,
  "loss_scale_window": 1000,
  "initial_scale_power": 16,
  "hysteresis": 2,
  "min_loss_scale": 1
},
"bf16": {
  "enabled": "auto"
}
```

FP16 和 BF16 的启用由训练参数自动决定。FP16 配置了动态损失缩放策略（初始缩放 2^16，窗口 1000 步）。

### 3. ZeRO Stage 2 优化

```json
"zero_optimization": {
  "stage": 2,
  "overlap_comm": false,
  "contiguous_gradients": true,
  "sub_group_size": 1e9,
  "reduce_bucket_size": 1e8
}
```

- `stage: 2`：分片优化器状态和梯度。
- `overlap_comm: false`：不重叠通信与计算（更稳定）。
- `contiguous_gradients: true`：使用连续梯度缓冲区提高效率。
- `reduce_bucket_size: 1e8`：梯度规约的桶大小（约 100M 元素）。

## 配置参数表格

| 参数名 | 值 | 说明 |
|--------|------|------|
| `stage` | `2` | ZeRO 优化阶段 |
| `overlap_comm` | `false` | 不重叠通信与计算 |
| `contiguous_gradients` | `true` | 使用连续梯度 |
| `sub_group_size` | `1e9` | 参数子组大小 |
| `reduce_bucket_size` | `1e8` | 梯度规约桶大小 |
| `load_universal` | `false` | 不使用通用检查点加载 |
| `zero_allow_untested_optimizer` | `true` | 允许未测试的优化器 |

## 与其他配置/模块的关系

- **被引用**：通过 `conf.yaml` 的 `training_args.deepspeed` 字段指定路径来使用。
- **与 zero2_offload.json 对比**：不包含 CPU 卸载，适合 GPU 显存充足的场景。
- **与 zero3.json 对比**：ZeRO-2 不分片模型参数，通信开销更低，但显存节省不如 ZeRO-3。

## 总结

`zero2.json` 提供了一个平衡的分布式训练配置。ZeRO Stage 2 在多 GPU 训练中分片优化器状态和梯度，在显存节省和通信效率之间取得良好平衡。适用于模型可以完整放入单 GPU 但优化器状态占用过大的场景，是 DreamZero 训练的推荐起点配置。
