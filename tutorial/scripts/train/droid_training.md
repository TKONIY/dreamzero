# DROID 训练脚本说明

> **注意**：源文件 `scripts/train/droid_training.sh` 不存在于当前代码库中。

## 实际可用的 DROID 训练脚本

DreamZero 提供了以下两个 DROID 数据集训练脚本：

- [droid_training_lora.md](droid_training_lora.md) -- LoRA 微调训练脚本（`scripts/train/droid_training_lora.sh`），参数高效微调，适合快速实验
- [droid_training_full_finetune.md](droid_training_full_finetune.md) -- 全参数微调训练脚本（`scripts/train/droid_training_full_finetune.sh`），更新全部模型参数

推荐从 LoRA 训练脚本开始了解 DreamZero 的训练流程。
