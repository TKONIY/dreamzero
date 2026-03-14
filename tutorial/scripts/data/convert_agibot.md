# DreamZero AGIBot 数据转换脚本详解

> 对应文件：`scripts/data/convert_agibot.py`

## 文件概述

`convert_agibot.py` 将 AGIBot 原始数据集（HDF5 + 视频格式）转换为 DreamZero 训练所需的 LeRobot v2 格式。该脚本支持 AGIBot 数据的两种目录结构格式（旧格式和新格式），并自动检测数据格式。

AGIBot 是一种双臂人形机器人，状态空间为 20 维，动作空间为 22 维，配备 3 个相机视角。

## 关键代码解析

### 1. modality 配置定义

```python
def generate_modality_json(output_dir: str) -> None:
    modality_config = {
        "state": {
            "left_arm_joint_position":  {"start": 0,  "end": 7,  ...},
            "right_arm_joint_position": {"start": 7,  "end": 14, ...},
            "left_effector_position":   {"start": 14, "end": 15, ...},
            "right_effector_position":  {"start": 15, "end": 16, ...},
            "head_position":            {"start": 16, "end": 18, ...},
            "waist_pitch":              {"start": 18, "end": 19, ...},
            "waist_lift":               {"start": 19, "end": 20, ...},
        },
        "action": {
            # 前 20 维与 state 相同
            "robot_velocity":           {"start": 20, "end": 22, ...},
        },
        ...
    }
```

AGIBot 的状态/动作空间结构清晰地定义了各关节的索引范围：

**状态（20 维）：**

| 子字段 | 维度范围 | 维数 | 说明 |
|--------|---------|------|------|
| `left_arm_joint_position` | [0, 7) | 7 | 左臂关节位置 |
| `right_arm_joint_position` | [7, 14) | 7 | 右臂关节位置 |
| `left_effector_position` | [14, 15) | 1 | 左夹爪位置 |
| `right_effector_position` | [15, 16) | 1 | 右夹爪位置 |
| `head_position` | [16, 18) | 2 | 头部位置 |
| `waist_pitch` | [18, 19) | 1 | 腰部俯仰 |
| `waist_lift` | [19, 20) | 1 | 腰部升降 |

**动作（22 维）** = 状态 20 维 + 机器人速度 2 维

### 2. 特征定义（FEATURES）

```python
FEATURES = {
    "observation.images.top_head": {"dtype": "video", "shape": [480, 640, 3], ...},
    "observation.images.hand_left": {"dtype": "video", ...},
    "observation.images.hand_right": {"dtype": "video", ...},
    "observation.state": {"dtype": "float32", "shape": [20]},
    "action": {"dtype": "float32", "shape": [22]},
    "annotation.language.action_text": {"dtype": "int64", "shape": [1]},
    ...
}
```

该字典定义了 LeRobot 数据集的特征模式，作为 `AgiBotDataset.create()` 的参数传入。

### 3. 数据格式自动检测

```python
def detect_dataset_format(src_path: str) -> str:
    if (src_path / "task_info").exists() and (src_path / "proprio_stats").exists():
        return "old"
    # 检查 job_id/robot_id/episode_id 结构
    for job_dir in subdirs:
        for robot_dir in job_dir.iterdir():
            for episode_dir in robot_dir.iterdir():
                if (episode_dir / "aligned_joints.h5").exists():
                    return "new"
    return "unknown"
```

**旧格式**特征：包含 `task_info/` 和 `proprio_stats/` 目录
**新格式**特征：嵌套的 `job_id/robot_id/episode_id` 结构，每个 episode 含 `aligned_joints.h5`

### 4. 旧格式数据加载

```python
def load_local_dataset_old_format(episode_id, src_path, task_id):
    with h5py.File(proprio_dir / "proprio_stats.h5") as f:
        state_joint = np.array(f["state/joint/position"])
        state_effector = np.clip(
            (np.array(f["state/effector/position"]) - 35.0) / (120.0 - 35.0), 0.0, 1.0
        )
        # ...
```

从 HDF5 文件中读取状态和动作数据。夹爪位置（effector）从原始范围 [35, 120] 归一化到 [0, 1]。

**旧格式目录结构**：
```
src_path/
├── task_info/task_{task_id}.json
├── proprio_stats/{task_id}/{episode_id}/proprio_stats.h5
└── observations/{task_id}/{episode_id}/videos/
    ├── head_color.mp4
    ├── hand_left_color.mp4
    └── hand_right_color.mp4
```

### 5. 新格式数据加载

```python
def load_local_dataset_new_format(episode_path):
    with h5py.File(joints_path) as f:
        state_left_effector = np.array(f["state/left_effector/position"])
        state_right_effector = np.array(f["state/right_effector/position"])
        state_effector = np.clip(
            np.column_stack([...]) - 35.0, 0.0, 85.0
        ) / 85.0
```

新格式将左右夹爪分开存储（而非合并），归一化方式也略有不同（先减 35 再除以 85）。

**新格式目录结构**：
```
src_path/
└── {job_id}/{robot_id}/{episode_id}/
    ├── aligned_joints.h5
    ├── data_info.json
    ├── head_color.mp4
    ├── hand_left_color.mp4
    └── hand_right_color.mp4
```

### 6. AgiBotDataset 类

```python
class AgiBotDataset(LeRobotDataset):
    def save_episode(self, task, episode_data=None, videos=None):
        # 自定义视频复制逻辑
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            shutil.copyfile(str(videos[key]), str(video_path))
```

继承 LeRobot 的 `LeRobotDataset`，重写了 `save_episode()` 方法以支持直接复制 MP4 文件（而非重新编码），重写了 `add_frame()` 去掉了一些不需要的校验，并自定义了 `consolidate()` 来使用自定义的统计量计算。

### 7. 任务和动作文本处理

```python
def create_tasks_jsonl(tgt_path, repo_id, task_name, all_action_texts):
    tasks = [task_name]  # 主任务名排第一
    tasks.extend(sorted(unique_action_texts))
```

AGIBot 数据集同时包含：
- **主任务名**（如 "组装零件"）：作为 task_index 0
- **动作文本**（如 "拿起螺丝"、"放置零件"）：从 `data_info.json` 或 `task_{id}.json` 中的 `action_config` 提取

每帧的 `annotation.language.action_text` 存储当前帧对应的动作文本索引。

### 8. 分块处理和内存管理

```python
for chunk_start in tqdm(range(0, len(all_subdir_eids), chunk_size), desc="Processing chunks"):
    # 处理 chunk_size 个 episode
    raw_datasets_chunk = process_map(load_local_dataset, ...)
    # 写入数据
    gc.collect()  # 显式垃圾回收
```

脚本将 episode 分成小块处理（默认每块 10 个），每块处理完后释放内存。这是处理大规模数据集时避免内存溢出的关键策略。

## 核心类/函数表格

| 类/函数 | 功能 |
|---------|------|
| `AgiBotDataset` | 继承 LeRobotDataset，定制 episode 保存和统计计算 |
| `main()` | 主转换流程，协调格式检测、数据加载和写入 |
| `detect_dataset_format()` | 自动检测旧格式 vs 新格式 |
| `load_local_dataset_old_format()` | 加载旧格式 AGIBot 数据 |
| `load_local_dataset_new_format()` | 加载新格式 AGIBot 数据 |
| `generate_modality_json()` | 生成 AGIBot 专用的 modality.json |
| `create_tasks_jsonl()` | 创建任务和动作文本的索引文件 |
| `compute_stats()` | 计算数据集统计量（均值、标准差等） |
| `get_stats_einops_patterns()` | 为不同数据类型确定聚合模式 |
| `get_task_instruction_old_format()` / `get_task_instruction_new_format()` | 提取任务语言指令 |

## 与其他模块的关系

- **训练脚本**：转换后的数据被 `scripts/train/agibot_training.sh` 使用
- **LeRobot 依赖**：基于 LeRobot 库的 `LeRobotDataset` 类进行数据写入
- **通用转换**：转换后可进一步使用 `convert_lerobot_to_gear.py` 生成额外的元数据（如 `relative_stats_dreamzero.json`）
- **数据配置**：需要在 `groot/vla/configs/data/dreamzero/` 下创建对应的 YAML 配置

## 使用示例

```bash
# 旧格式（需要 task_id）
python scripts/data/convert_agibot.py \
    --src_path ./raw_data/agibot_old \
    --task_id 3222 \
    --tgt_path ./data/agibot_lerobot

# 新格式（自动检测）
python scripts/data/convert_agibot.py \
    --src_path ./raw_data/agibot_new \
    --tgt_path ./data/agibot_lerobot

# 调试模式（仅处理 2 个 episode）
python scripts/data/convert_agibot.py \
    --src_path ./raw_data/agibot_new \
    --tgt_path ./data/agibot_lerobot \
    --debug
```

## 总结

`convert_agibot.py` 是 AGIBot 数据到 LeRobot 格式的完整转换管线。脚本的亮点包括：自动检测新旧两种数据格式、继承 LeRobotDataset 定制保存逻辑、分块处理避免内存溢出、以及完整的任务/动作文本索引系统。理解该脚本中的状态/动作维度定义和夹爪归一化逻辑，对于在 AGIBot 平台上训练和调试 DreamZero 模型至关重要。
