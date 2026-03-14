# zero2_offload.json -- DeepSpeed ZeRO Stage 2 + CPU 卸载配置

## 文件概述

`zero2_offload.json` 是在 ZeRO Stage 2 基础上增加了 **CPU 卸载（Offload）** 功能的配置。通过将优化器状态卸载到 CPU 内存，进一步降低 GPU 显存占用，适合 GPU 显存较紧张但 CPU 内存充足的训练场景。

## 关键配置项解析

### 1. CPU 卸载配置

```json
"zero_optimization": {
  "stage": 2,
  "offload_optimizer": {
    "device": "cpu",
    "pin_memory": true
  },
  ...
}
```

关键新增项：
- `offload_optimizer.device: "cpu"`：将优化器状态（如 Adam 的一阶和二阶动量）卸载到 CPU。
- `pin_memory: true`：使用锁页内存加速 CPU-GPU 数据传输。

### 2. 其余配置

与 `zero2.json` 完全一致，包括自动批次配置、混合精度设置、`overlap_comm: false`、`contiguous_gradients: true` 等。

## 配置参数表格

| 参数名 | 值 | 说明 |
|--------|------|------|
| `stage` | `2` | ZeRO 优化阶段 |
| `offload_optimizer.device` | `"cpu"` | 优化器卸载到 CPU |
| `offload_optimizer.pin_memory` | `true` | 使用锁页内存 |
| `overlap_comm` | `false` | 不重叠通信与计算 |
| `contiguous_gradients` | `true` | 使用连续梯度 |
| `sub_group_size` | `1e9` | 参数子组大小 |
| `reduce_bucket_size` | `1e8` | 梯度规约桶大小 |

## 与其他配置/模块的关系

- **基于**：`zero2.json`，增加了 `offload_optimizer` 配置块。
- **与 zero2.json 对比**：增加了 CPU 卸载，进一步节省 GPU 显存，但会增加 CPU-GPU 数据传输开销，降低训练速度。
- **与 zero3.json 对比**：ZeRO-2 + 卸载通常比 ZeRO-3 的通信开销低，适合作为 GPU 显存不足时的折中方案。

## 总结

`zero2_offload.json` 是 `zero2.json` 的显存优化变体。通过将优化器状态卸载到 CPU 内存并使用锁页内存加速传输，可以在 GPU 显存紧张时继续训练大模型。权衡是训练速度会因 CPU-GPU 数据传输而略有下降。建议在 GPU 显存不足以使用纯 `zero2.json` 时选用此配置。
