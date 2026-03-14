# DreamZero 损失曲线对比脚本详解

> 对应文件：`scripts/compare_loss.py`

## 文件概述

`compare_loss.py` 是一个轻量级的训练分析工具，用于对比 LoRA 微调和全参数微调两种训练策略的损失曲线。脚本读取训练过程中产生的 `loss_log.jsonl` 日志文件，输出对比表格和可视化图表。

该脚本是 DreamZero 实验分析流程的重要组成部分，帮助研究者直观比较不同训练策略的收敛速度和最终性能。

## 关键代码解析

### 1. 日志文件加载

```python
def load_loss_log(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
```

读取 JSONL 格式的日志文件，每行是一个 JSON 对象，包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `step` | int | 训练步数 |
| `loss` | float | 总损失 |
| `dynamics_loss_avg` | float | 动态预测损失（视频生成） |
| `action_loss_avg` | float | 动作预测损失 |

### 2. 对比表格输出

```python
def print_comparison_table(lora_entries, full_entries):
    lora_by_step = {e["step"]: e for e in lora_entries}
    full_by_step = {e["step"]: e for e in full_entries}
    all_steps = sorted(set(lora_by_step.keys()) | set(full_by_step.keys()))
```

按训练步数对齐两组日志，输出如下格式的对比表格：

```
  Step   LoRA Loss  Full Loss   LoRA Dyn   Full Dyn   LoRA Act   Full Act
------  ----------  ----------  ----------  ----------  ----------  ----------
    10      0.5234      0.4987      0.3012      0.2856      0.2222      0.2131
    20      0.4567      0.4234      ...
```

当某一方在某步没有记录时，显示破折号 `--`。

### 3. 可视化图表

```python
def plot_comparison(lora_entries, full_entries, output_path):
    metrics = [
        ("loss", "Total Loss"),
        ("dynamics_loss_avg", "Dynamics Loss"),
        ("action_loss_avg", "Action Loss"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
```

生成三个子图的对比图表：

1. **Total Loss**：总损失曲线对比
2. **Dynamics Loss**：动态预测损失（反映视频生成质量）
3. **Action Loss**：动作预测损失（反映控制精度）

每个子图中，LoRA 用圆形标记、全参数微调用方形标记，便于区分。图表以 150 DPI 保存为 PNG 文件。

### 4. 命令行接口

```python
parser.add_argument("--lora-log", required=True)
parser.add_argument("--full-log", required=True)
parser.add_argument("--plot", default=None)
```

`--plot` 为可选参数，不指定时只输出文本表格，指定后额外生成可视化图表。

## 核心类/函数表格

| 函数 | 功能 |
|------|------|
| `load_loss_log(path)` | 从 JSONL 文件加载损失日志 |
| `print_comparison_table(lora, full)` | 按步数对齐输出对比表格 |
| `plot_comparison(lora, full, output)` | 使用 matplotlib 生成三面板对比图 |
| `main()` | 解析参数并调度表格/图表生成 |

## 与其他模块的关系

- **训练脚本**：`scripts/train/droid_training_lora.sh` 和 `droid_training_full_finetune.sh` 训练时生成 `loss_log.jsonl`
- **训练入口**：`groot/vla/experiment/experiment.py` 中的训练循环负责写入损失日志
- **DreamZero 损失**：总损失 = 动态损失（dynamics_loss）+ 动作损失（action_loss），分别对应视频生成和动作预测两个目标

## 使用示例

```bash
# 仅输出文本表格
python scripts/compare_loss.py \
    --lora-log ./checkpoints/dreamzero_droid_lora/loss_log.jsonl \
    --full-log ./checkpoints/dreamzero_droid_full_finetune/loss_log.jsonl

# 输出表格 + 可视化图表
python scripts/compare_loss.py \
    --lora-log ./checkpoints/dreamzero_droid_lora/loss_log.jsonl \
    --full-log ./checkpoints/dreamzero_droid_full_finetune/loss_log.jsonl \
    --plot loss_comparison.png
```

## 总结

`compare_loss.py` 提供了简洁的 LoRA vs 全参数微调损失对比功能。脚本将损失分解为动态预测损失和动作预测损失两个维度进行对比分析，既支持命令行文本表格（快速查看），也支持 matplotlib 可视化图表（论文和报告用）。它是 DreamZero 训练实验分析工具链中不可或缺的一环。
