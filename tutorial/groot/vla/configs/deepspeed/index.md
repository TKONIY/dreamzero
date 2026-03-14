# DeepSpeed 分布式训练配置索引

## 概述

`deepspeed/` 目录包含 DeepSpeed ZeRO（Zero Redundancy Optimizer）的分布式训练配置文件。DreamZero 提供了三种 ZeRO 优化策略，用户可根据 GPU 显存和通信带宽情况选择合适的配置。

## 文件索引

- [zero2.json](zero2.md) -- ZeRO Stage 2（分片优化器状态+梯度）
- [zero2_offload.json](zero2_offload.md) -- ZeRO Stage 2 + CPU 卸载（额外将优化器卸载到 CPU）
- [zero3.json](zero3.md) -- ZeRO Stage 3（分片模型参数+梯度+优化器状态）

## 配置对比

| 特性 | ZeRO-2 | ZeRO-2 + Offload | ZeRO-3 |
|-----|--------|------------------|--------|
| 优化器状态分片 | 是 | 是（+ CPU 卸载） | 是 |
| 梯度分片 | 是 | 是 | 是 |
| 模型参数分片 | 否 | 否 | 是 |
| 通信重叠 | 否 | 否 | 是 |
| GPU 显存节省 | 中等 | 较高 | 最高 |
| 训练速度 | 最快 | 较慢 | 依赖互联带宽 |
| 推荐场景 | GPU 显存充足 | GPU 显存紧张 | 超大模型 |

## 选择建议

1. **优先使用 `zero2.json`**：通信开销最低，训练速度最快。适用于模型参数可放入单 GPU 的场景。
2. **显存不足时用 `zero2_offload.json`**：将优化器卸载到 CPU，以训练速度换取显存空间。
3. **超大模型用 `zero3.json`**：当模型参数也无法放入单 GPU 时使用。需要高带宽 GPU 互联（如 NVLink）以减少通信瓶颈。

## 使用方式

通过命令行指定 DeepSpeed 配置文件路径：

```bash
python train.py \
  training_args.deepspeed=groot/vla/configs/deepspeed/zero2.json \
  ...
```
