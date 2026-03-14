# zero3.json -- DeepSpeed ZeRO Stage 3 配置

## 文件概述

`zero3.json` 定义了 DeepSpeed ZeRO Stage 3 的分布式训练配置。ZeRO-3 是最激进的显存优化策略，在多 GPU 之间分片**模型参数、梯度和优化器状态**三者，实现最大程度的显存节省。适用于超大模型无法在单 GPU 上放下的场景。

## 关键配置项解析

### 1. ZeRO Stage 3 优化

```json
"zero_optimization": {
  "stage": 3,
  "overlap_comm": true,
  "contiguous_gradients": true,
  "sub_group_size": 1e9,
  "reduce_bucket_size": 5e8,
  "stage3_prefetch_bucket_size": 5e8,
  "stage3_param_persistence_threshold": 1e6,
  "stage3_max_live_parameters": 1e9,
  "stage3_max_reuse_distance": 1e9,
  "stage3_gather_16bit_weights_on_model_save": true
}
```

关键参数解析：
- `overlap_comm: true`：与 ZeRO-2 不同，ZeRO-3 启用了通信与计算重叠以减少延迟。
- `reduce_bucket_size: 5e8`：更大的规约桶（500M 元素），适应更大的通信量。
- `stage3_prefetch_bucket_size: 5e8`：参数预取桶大小，提前从其他 GPU 收集下一步需要的参数。
- `stage3_param_persistence_threshold: 1e6`：小于 100 万参数的层保留在每个 GPU 上（不分片），避免对小参数频繁通信。
- `stage3_max_live_parameters: 1e9`：同时在 GPU 上保持的最大参数量。
- `stage3_max_reuse_distance: 1e9`：参数重用距离阈值。
- `stage3_gather_16bit_weights_on_model_save: true`：保存模型时在主节点收集所有 16 位权重，确保检查点完整可用。

## 配置参数表格

| 参数名 | 值 | 说明 |
|--------|------|------|
| `stage` | `3` | ZeRO 优化阶段（最高级别） |
| `overlap_comm` | `true` | 重叠通信与计算 |
| `contiguous_gradients` | `true` | 使用连续梯度 |
| `reduce_bucket_size` | `5e8` | 梯度规约桶大小 |
| `stage3_prefetch_bucket_size` | `5e8` | 参数预取桶大小 |
| `stage3_param_persistence_threshold` | `1e6` | 小参数保留阈值 |
| `stage3_max_live_parameters` | `1e9` | GPU 上最大活跃参数量 |
| `stage3_max_reuse_distance` | `1e9` | 参数重用距离 |
| `stage3_gather_16bit_weights_on_model_save` | `true` | 保存时收集完整权重 |

## 与其他配置/模块的关系

- **与 zero2.json 对比**：ZeRO-3 额外分片模型参数，显存节省最大但通信开销也最大。
- **与 zero2_offload.json 对比**：ZeRO-3 不使用 CPU 卸载，所有数据保持在 GPU 上但分布在多 GPU 之间。
- **检查点保存**：`stage3_gather_16bit_weights_on_model_save: true` 确保保存的检查点可以在非 ZeRO-3 环境下加载。

## 总结

`zero3.json` 是 DreamZero 提供的最高级别显存优化配置。ZeRO Stage 3 通过在所有 GPU 之间完全分片模型参数、梯度和优化器状态，使得训练超大模型成为可能。代价是更高的通信开销，因此需要高带宽的 GPU 互联（如 NVLink）。适用于单 GPU 显存无法容纳模型参数的极端场景，例如使用完整 WAN 模型而非 LoRA 微调时。
