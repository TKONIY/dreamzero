# 初始动作提取工具 (`groot/vla/data/conversion/gr1/get_initial_actions.py`)

## 文件概述

`get_initial_actions.py` 提供了从 GR1 数据集中提取、保存和加载每条轨迹初始动作的工具函数。初始动作是指每条轨迹第一步的动作值，在相对动作计算和数据预处理中有重要用途。该文件支持两种数据格式：旧版 HDF5 格式和新版 LeRobot 格式。

## 关键代码解析

### 1. `get_initial_actions` 函数（HDF5 格式）

从旧版 HDF5 格式数据中提取初始动作：

```python
def get_initial_actions(data_dir):
    hdf5_file = h5py.File(Path(data_dir) / TRAINABLE_HDF5_FILENAME, "r")
    initial_actions = {}
    for demo_name in hdf5_file["data"].keys():
        demo_group = hdf5_file["data"][demo_name]
        initial_actions[demo_name] = {}
        for action_key in demo_group["action"].keys():
            initial_actions[demo_name][action_key] = demo_group["action"][action_key][0]
    return [initial_actions]
```

返回格式为 `[{trajectory_name: {action_key: np.ndarray}}]`。

### 2. `get_initial_actions_from_lerobot` 函数（LeRobot 格式）

从 LeRobot 格式数据中提取初始动作，需要处理模态切片：

```python
def get_initial_actions_from_lerobot(data_dir):
    # 1. 加载模态元数据，获取动作键定义
    meta_modality = U.load_json(meta_modality_path)
    action_keys = meta_modality["action"].keys()

    # 2. 获取数据路径模式和 chunk 大小
    data_path_pattern = meta_info["data_path"]
    chunk_size = meta_info["chunks_size"]

    # 3. 遍历所有剧集
    for episode_info in episode_metadata:
        episode_path = data_dir / data_path_pattern.format(...)
        episode_data = pd.read_parquet(episode_path)

        # 4. 从拼接的动作向量中按 start:end 切片
        initial_action_concat = episode_data["action"].iloc[0]
        for action_key in action_keys:
            start = meta_modality["action"][action_key]["start"]
            end = meta_modality["action"][action_key]["end"]
            initial_actions[trajectory_id][action_key] = initial_action_concat[start:end]
```

这里体现了 LeRobot 格式的核心设计：动作被拼接为单个向量存储在 `action` 列中，需要通过 `start`/`end` 索引切片还原各分量。

### 3. `save_initial_actions` 函数

使用 numpy 的 npz 格式保存初始动作：

```python
def save_initial_actions(initial_actions, initial_actions_path):
    np.savez(str(initial_actions_path), initial_actions)
```

### 4. `load_initial_actions` 函数

加载已保存的初始动作，处理 npz 文件的嵌套结构：

```python
def load_initial_actions(initial_actions_path):
    initial_actions_npz = np.load(str(initial_actions_path), allow_pickle=True)
    initial_actions_array = initial_actions_npz["arr_0"]
    # 遍历多数据集结构
    for dataset_initial_actions in initial_actions_array:
        for trajectory_name, action_dict in dataset_initial_actions.items():
            initial_actions_for_this_dataset[trajectory_name] = action_dict
```

返回格式为 `list[dict[str, dict[str, np.ndarray]]]`，外层列表对应多个数据集。

### 5. 命令行入口

文件可作为独立脚本运行：

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()
    initial_actions = get_initial_actions_from_lerobot(args.data_dir)
    save_initial_actions(initial_actions, ...)
```

用法：`python get_initial_actions.py /path/to/lerobot/dataset`

## 核心类/函数表格

| 名称 | 类型 | 说明 |
|------|------|------|
| `get_initial_actions` | 函数 | 从 HDF5 格式提取初始动作 |
| `get_initial_actions_from_lerobot` | 函数 | 从 LeRobot 格式提取初始动作 |
| `save_initial_actions` | 函数 | 将初始动作保存为 npz 文件 |
| `load_initial_actions` | 函数 | 从 npz 文件加载初始动作 |

## 与其他模块的关系

- **依赖 `constants.py`** - 使用 `TRAINABLE_HDF5_FILENAME`、`INITIAL_ACTIONS_FILENAME` 等常量
- **依赖 `dataset/macro.py`** - 使用 LeRobot 元数据文件名常量
- **依赖 `common.utils`** - 使用 `load_json` 和 `load_jsonl` 工具函数
- **被 `dataset/lerobot.py` 调用** - `load_initial_actions` 被数据集初始化时使用

## 总结

`get_initial_actions.py` 是数据转换管线中的辅助工具，负责提取每条轨迹的第一步动作。它支持从旧版 HDF5 和新版 LeRobot 两种格式中提取数据，并通过 npz 文件进行持久化存储。在 LeRobot 格式中，需要使用模态元数据中的 `start`/`end` 索引对拼接动作向量进行切片，这与 `schema/lerobot.py` 中 `LeRobotStateActionMetadata` 的设计完全一致。
