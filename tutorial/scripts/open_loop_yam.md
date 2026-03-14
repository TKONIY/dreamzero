# DreamZero YAM 开环评估脚本详解

> 对应文件：`scripts/open_loop_yam.py`

## 文件概述

`open_loop_yam.py` 是 DreamZero 在 YAM 数据集上的离线开环评估（Open-Loop Evaluation）脚本。该脚本直接加载模型检查点（无需部署推理服务），逐帧读取 YAM 数据集，运行模型推理，将预测动作与真实（Ground Truth）动作进行对比，计算 MSE（均方误差）指标并生成可视化图表。

"开环"意味着模型的预测结果不会反馈到环境中影响后续观测，每一帧的输入都来自真实数据。这是评估模型动作预测精度的标准方法。

## 关键代码解析

### 1. TorchDynamo 禁用

```python
import torch._dynamo
torch._dynamo.config.disable = True
```

禁用 PyTorch 2.0 的 `torch.compile` 动态编译器，确保推理时的行为一致性和调试友好性。

### 2. YAM 数据布局定义

```python
VIDEO_CAMERAS = {
    "video.top_camera-images-rgb":   "observation.images.top_camera-images-rgb",
    "video.left_camera-images-rgb":  "observation.images.left_camera-images-rgb",
    "video.right_camera-images-rgb": "observation.images.right_camera-images-rgb",
}

STATE_SLICES = {
    "state.left_joint_pos":    (34, 40),
    "state.left_gripper_pos":  (32, 33),
    "state.right_joint_pos":   (40, 46),
    "state.right_gripper_pos": (33, 34),
}

ACTION_SLICES = {
    "action.left_joint_pos":    (34, 40),
    "action.left_gripper_pos":  (32, 33),
    "action.right_joint_pos":   (40, 46),
    "action.right_gripper_pos": (33, 34),
}
```

YAM 数据的状态和动作是从更大维度的向量中按索引切片提取的。注意索引范围不是从 0 开始的（如 `left_joint_pos` 从第 34 维开始），说明原始向量中包含了其他不使用的字段。

### 3. YAMDataset 类

```python
class YAMDataset:
    def __init__(self, dataset_path: str):
        parquet_files = sorted(glob.glob(...))
        self.episodes = []
        self.cum_lengths = [0]
        for pf in parquet_files:
            table = pq.read_table(pf)
            self.episodes.append(table)
            self.cum_lengths.append(self.cum_lengths[-1] + table.num_rows)
```

轻量级数据集读取器，特点包括：
- 加载所有 Parquet 文件到内存中的 PyArrow Table
- 使用累积长度索引（`cum_lengths`）实现全局行索引到 episode + 帧的映射
- 通过 OpenCV 按帧号随机访问 MP4 视频

```python
def get_frame(self, idx, server_key) -> np.ndarray:
    cap = cv2.VideoCapture(mp4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, row)
    ret, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

每次读取单帧后立即释放视频句柄，避免文件描述符泄漏。

### 4. 观测构建

```python
def build_obs(dataset: YAMDataset, idx: int, prompt: str) -> dict:
    obs = {}
    for server_key in dataset.video_dirs:
        frame = dataset.get_frame(idx, server_key)
        obs[server_key] = frame[np.newaxis, ...].astype(np.uint8)  # (1, H, W, C)

    state = dataset.get_state(idx)
    for key, (start, end) in STATE_SLICES.items():
        obs[key] = state[start:end].reshape(1, -1)  # (1, D)

    obs["annotation.task"] = prompt
    return obs
```

构建符合 `GrootSimPolicy.forward()` 输入格式的观测字典：
- 图像帧增加 batch 维度 → `(1, H, W, C)`
- 状态切片增加 batch 维度 → `(1, D)`
- 添加文本提示（任务描述）

### 5. 推理循环

```python
def evaluate(args):
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", world_size=1, rank=0)

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.YAM,
        model_path=args.model_path,
        device=args.device,
    )
```

**分布式初始化**：`GrootSimPolicy` 内部使用 `dist.get_rank()`，因此即使单 GPU 推理也需要初始化分布式环境（使用 `gloo` 后端）。

**推理调用**：
```python
with torch.inference_mode():
    result, _ = policy.lazy_joint_forward_causal(Batch(obs=obs))
```

使用 `lazy_joint_forward_causal` 进行因果推理。`Batch` 来自 `tianshou` 库，用于封装观测数据。

### 6. 结果收集和指标计算

```python
for k in ACTION_KEY_ORDER:
    if k in result.act:
        pred_val = result.act[k]
        pred_val = np.atleast_1d(pred_val[0]).flatten()
        preds_per_key[k].append(pred_val)
        gts_per_key[k].append(gt[k])
```

对每个动作子键（左关节、左夹爪、右关节、右夹爪），分别收集预测值和真实值。取预测结果的第一个时间步（`pred_val[0]`），与当前帧的真实动作对比。

### 7. 可视化输出

```python
def save_plots(all_preds, all_gts, key_names, output_dir):
```

生成三类可视化图表：

1. **逐维度对比图**（`action_dim_*.png`）：每个动作维度的预测 vs 真实曲线，标注该维度的 MSE
2. **全维度汇总图**（`all_action_dims.png`）：所有维度在一张图中展示，便于整体观察
3. **按键汇总图**（`per_key_summary.png`）：按动作子键（左关节、右关节等）分组展示，实线为预测、虚线为真实

### 8. 结果保存

```python
with open(os.path.join(args.output_dir, "mse.txt"), "w") as f:
    f.write(f"overall_mse,{overall_mse}\n")
    for k in valid_keys:
        k_mse = float(np.mean((stacked_preds[k] - stacked_gts[k]) ** 2))
        f.write(f"{k},{k_mse}\n")
    for d, v in enumerate(mse_dim):
        f.write(f"dim_{d},{v}\n")
```

将 MSE 结果保存为 CSV 格式的文本文件，包含总体 MSE、按键 MSE 和按维度 MSE。

## 核心类/函数表格

| 类/函数 | 功能 |
|---------|------|
| `YAMDataset` | 轻量级 YAM 数据集读取器，支持 Parquet + MP4 |
| `build_obs()` | 从数据集构建模型输入观测字典 |
| `get_gt_action_dict()` | 提取真实动作并按子键拆分 |
| `evaluate()` | 主评估循环：加载模型、逐帧推理、收集结果 |
| `save_plots()` | 生成三类可视化对比图表 |
| `GrootSimPolicy` | DreamZero 推理策略（来自 `groot.vla.model.n1_5.sim_policy`） |
| `EmbodimentTag.YAM` | YAM 机器人形态标签（来自 `groot.vla.data.schema`） |

## 与其他模块的关系

- **模型来源**：评估的模型检查点由 `scripts/train/yam_training.sh` 训练产生
- **推理策略**：`groot/vla/model/n1_5/sim_policy.py` 中的 `GrootSimPolicy` 封装了完整的推理流程
- **数据格式**：读取 LeRobot v2 格式的 YAM 数据集（由数据转换脚本生成）
- **Embodiment 标签**：`groot/vla/data/schema.py` 中定义了 `EmbodimentTag.YAM`

## 使用示例

```bash
# 基本用法
python scripts/open_loop_yam.py \
    --model_path ./checkpoints/dreamzero_yam_lora/checkpoint-10000 \
    --dataset_path ./data/yam_lerobot \
    --device cuda:0 \
    --num_samples 200

# 使用数据集中的任务描述作为提示
python scripts/open_loop_yam.py \
    --model_path ./checkpoints/dreamzero_yam_lora/checkpoint-10000 \
    --dataset_path ./data/yam_lerobot \
    --use_dataset_prompt \
    --num_samples 300 \
    --output_dir results_yam_eval

# 从特定帧开始评估
python scripts/open_loop_yam.py \
    --model_path ./checkpoints/dreamzero_yam_lora/checkpoint-10000 \
    --dataset_path ./data/yam_lerobot \
    --start_idx 500 \
    --num_samples 100
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | (必须) | 模型检查点路径 |
| `--dataset_path` | (必须) | YAM 数据集根目录 |
| `--device` | `cuda:0` | 推理设备 |
| `--prompt` | `"pick up the object"` | 默认任务提示 |
| `--use_dataset_prompt` | `False` | 使用数据集中的标注作为提示 |
| `--num_samples` | 300 | 评估样本数 |
| `--start_idx` | 0 | 起始帧索引 |
| `--output_dir` | `results_yam` | 输出目录 |
| `--log_every` | 10 | 日志打印间隔 |

## 总结

`open_loop_yam.py` 是 DreamZero YAM 模型的离线开环评估工具。脚本直接加载检查点进行推理，无需部署服务，适合快速验证模型效果。核心输出包括逐维度的 MSE 指标和多种粒度的可视化图表。脚本中的 `YAMDataset` 类和 `build_obs()` 函数展示了如何从 LeRobot 格式数据构建模型输入，对于理解 DreamZero 的推理流程和开发新的评估脚本具有参考价值。
