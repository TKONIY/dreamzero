# DreamZero LeRobot 到 GEAR 格式转换脚本详解

> 对应文件：`scripts/data/convert_lerobot_to_gear.py`

## 文件概述

`convert_lerobot_to_gear.py` 是一个通用的元数据生成工具，将标准的 LeRobot v2 数据集增强为 DreamZero/GEAR 训练流水线所需的格式。与 `convert_droid.py` 和 `convert_agibot.py` 不同，该脚本 **不修改数据文件本身**（Parquet 和视频文件保持不变），仅创建或更新 `meta/` 目录下的元数据文件。

这使得它成为将任意自定义机器人数据集接入 DreamZero 训练的标准工具。

## 关键代码解析

### 1. 有效的 Embodiment 标签

```python
VALID_EMBODIMENT_TAGS = [
    "real_gr1_arms_only", "oxe_droid", "agibot", "yam", "xdof",
    "dream", "sim_behavior_r1_pro", ...
]
```

DreamZero 训练流水线通过 `embodiment_tag` 选择对应的数据配置和模型适配。`xdof` 是默认通用标签，适用于自定义机器人。脚本会验证用户提供的标签是否在有效列表中。

### 2. 特征自动检测

```python
def detect_features(info: dict) -> dict:
    features = info.get("features", {})
    state_keys = [k for k in features if k.startswith("observation.state")]
    action_keys = [k for k in features if k == "action" or k.startswith("action.")]
    video_keys = [k for k in features if features[k].get("dtype") == "video"]
    annotation_keys = [k for k in features if k.startswith("annotation")]
    return {"state": state_keys, "action": action_keys, "video": video_keys, "annotation": annotation_keys}
```

从 `info.json` 中自动识别各类特征：
- **状态**：以 `observation.state` 开头的列
- **动作**：`action` 或以 `action.` 开头的列
- **视频**：`dtype` 为 `"video"` 的特征
- **标注**：以 `annotation` 开头的列

### 3. modality.json 构建

```python
def build_modality_json(info, detected, state_mapping, action_mapping, task_key):
```

`modality.json` 是 DreamZero 数据加载的核心配置文件，定义了状态和动作向量中各子字段的映射关系。

**自动模式**（未指定 `--state-keys`/`--action-keys`）：
```python
modality["state"]["state"] = {
    "original_key": state_col,
    "start": 0, "end": dim,
    "rotation_type": None, "absolute": True,
}
```
将整个状态/动作向量作为单个字段处理。

**手动模式**（通过 JSON 指定子字段映射）：
```bash
--state-keys '{"joint_pos": [0, 6], "gripper_pos": [6, 7]}'
```
按用户指定的索引范围拆分为多个命名子字段，这对于相对动作统计和跨 embodiment 迁移学习非常重要。

### 4. 数据集统计量计算

```python
def compute_stats(parquet_paths, columns):
    for pp in tqdm(parquet_paths, desc="Computing stats"):
        df = pd.read_parquet(pp)
        for col in columns:
            arr = np.stack(df[col].values)
            all_data[col].append(arr)
    # 计算 mean, std, min, max, q01, q99
```

遍历所有 Parquet 文件，计算各数值列的统计量。这些统计量被训练流水线用于数据归一化。

### 5. 相对动作统计量

```python
def compute_relative_stats(parquet_paths, modality, relative_action_keys, action_horizon=24):
    for i in range(max(usable, 0)):
        ref_state = state_slice[i]
        chunk_end = min(i + action_horizon, traj_len)
        actions = action_slice[i:chunk_end]
        relative = actions - ref_state
        all_relative.extend(relative)
```

相对动作统计是 DreamZero 的特色功能。对于每个时间步 `i`，计算未来 `action_horizon`（默认 24）步动作相对于当前状态的差值。这模拟了训练时 `_calculate_relative_stats_for_key`（位于 `groot/vla/data/dataset/lerobot.py`）的逻辑。

计算结果保存为 `relative_stats_dreamzero.json`，用于相对动作空间的归一化。

### 6. 任务和 Episode 元数据

```python
def build_tasks(parquet_paths, task_key):
    # 从所有 episode 中提取唯一任务描述

def build_episodes(parquet_paths, info, task_key, tasks):
    # 为每个 episode 记录索引、长度和关联任务
```

生成 `tasks.jsonl`（唯一任务列表）和 `episodes.jsonl`（每个 episode 的元数据），供训练时的数据采样使用。

### 7. 数据集验证

```python
def validate_dataset(dataset_path, info, modality):
    # 检查必要目录（data/, videos/, meta/）
    # 检查视频特征
    # 检查状态/动作定义
    # 检查 episode 数和 FPS
```

转换完成后进行基本的完整性验证，报告潜在问题。

### 8. 主流程

`main()` 函数按以下顺序执行：

1. 加载 `info.json`，检测特征
2. 构建并写入 `modality.json`
3. 写入 `embodiment.json`
4. 计算并写入 `stats.json`
5. 计算并写入 `relative_stats_dreamzero.json`（可选）
6. 构建并写入 `tasks.jsonl`
7. 构建并写入 `episodes.jsonl`
8. 验证数据集完整性

所有步骤都支持 `--force` 覆盖已有文件，未使用 `--force` 时会跳过已存在的文件。

## 核心类/函数表格

| 函数 | 功能 |
|------|------|
| `main()` | 主流程：解析参数，协调所有转换步骤 |
| `load_info()` | 加载 `meta/info.json` |
| `detect_features()` | 从 info.json 自动检测特征分类 |
| `build_modality_json()` | 构建 modality.json 结构 |
| `parse_key_mapping()` | 解析用户提供的 JSON 键映射 |
| `compute_stats()` | 计算 mean/std/min/max/q01/q99 统计量 |
| `compute_relative_stats()` | 计算相对动作统计量 |
| `build_tasks()` | 从 Parquet 文件提取唯一任务列表 |
| `build_episodes()` | 构建 episode 级别元数据 |
| `validate_dataset()` | 验证数据集完整性 |
| `get_parquet_paths()` | 根据 info.json 中的模式获取所有 Parquet 路径 |

## 与其他模块的关系

- **上游数据**：可以处理由 `convert_droid.py` 或 `convert_agibot.py` 生成的数据，也可处理任何符合 LeRobot v2 标准的数据集
- **数据加载**：生成的元数据被 `groot/vla/data/dataset/lerobot.py` 使用
- **训练配置**：转换后需要在 `groot/vla/configs/data/dreamzero/` 下创建对应的 YAML 配置文件
- **文档参考**：完整的自定义训练流程见 `docs/CUSTOM_EMBODIMENT_TRAINING.md`

## 使用示例

```bash
# 自动检测，使用默认 xdof embodiment
python scripts/data/convert_lerobot_to_gear.py --dataset-path ./Dataset/my_robot_data

# 指定字段映射和 embodiment
python scripts/data/convert_lerobot_to_gear.py \
    --dataset-path ./Dataset/my_robot_data \
    --embodiment-tag yam \
    --state-keys '{"joint_pos": [0, 6], "gripper_pos": [6, 7]}' \
    --action-keys '{"joint_pos": [0, 6], "gripper_pos": [6, 7]}' \
    --relative-action-keys joint_pos \
    --task-key annotation.task

# 输出到新目录（不修改原始数据）
python scripts/data/convert_lerobot_to_gear.py \
    --dataset-path ./Dataset/my_robot_data \
    --output-path ./Dataset/my_robot_data_gear
```

## 总结

`convert_lerobot_to_gear.py` 是 DreamZero 数据流水线中最通用的工具，为任意 LeRobot v2 数据集生成 GEAR/DreamZero 训练所需的元数据。脚本不修改原始数据文件，仅创建 `modality.json`、`embodiment.json`、`stats.json`、`relative_stats_dreamzero.json`、`tasks.jsonl` 和 `episodes.jsonl`。支持自动特征检测和手动字段映射两种模式，适用于从简单的单臂机器人到复杂的多关节人形机器人等各种场景。
