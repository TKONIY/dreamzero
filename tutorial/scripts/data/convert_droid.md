# DreamZero DROID 数据转换脚本详解

> 对应文件：`scripts/data/convert_droid.py`

## 文件概述

`convert_droid.py` 将 DROID 1.0.1 数据集从原始的 RLDS/TFDS（TensorFlow Datasets）格式转换为 DreamZero 训练所需的 LeRobot v2 格式。转换过程包括：

1. 加载 RLDS 格式的 TFRecord 文件
2. 过滤失败的 episode 和缺少语言标注的 episode
3. 基于预计算的空闲帧范围进行空闲帧过滤（参考 Physical Intelligence 的 OpenPI 方法）
4. 将状态、动作数据写入 Parquet 文件
5. 将多相机视频编码为 MP4 文件
6. 生成完整的元数据文件

## 关键代码解析

### 1. 线程控制

```python
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
# ... 其他线程控制
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
```

脚本使用多进程并行处理（`ProcessPoolExecutor`），因此将每个进程内的线程数限制为 1，避免多进程与多线程的过度订阅（oversubscription），这是大规模数据处理中的常见最佳实践。

### 2. RLDS 数据广播

```python
def _broadcast_metadata_rlds(i: tf.Tensor, traj: dict) -> dict:
    steps = traj.pop("steps")
    traj_len = tf.shape(tf.nest.flatten(steps)[0])[0]
    metadata = tf.nest.map_structure(lambda x: tf.repeat(x, traj_len), traj)
    traj = {**steps, "traj_metadata": metadata}
    traj["_len"] = tf.repeat(traj_len, traj_len)
    traj["_traj_index"] = tf.repeat(i, traj_len)
    traj["_frame_index"] = tf.range(traj_len)
    return traj
```

RLDS 格式中，轨迹级别的元数据（如 episode metadata）需要广播到每一帧。该函数将 `traj` 中的元数据按轨迹长度重复，并添加帧索引、轨迹索引等辅助字段，使得每一帧都携带完整的上下文信息。

### 3. 状态/动作拼接

```python
def concat_state_or_action(modality_dict, keys, compute_concat_info=False):
```

该函数将分散存储在不同键下的状态或动作分量拼接为单一向量。例如，DROID 的状态包括：

- `cartesian_position`（6 维）
- `gripper_position`（1 维）
- `joint_position`（7 维）

拼接后生成 14 维状态向量，同时记录各分量在向量中的起止位置（`concat_info`），写入 `modality.json` 供训练时使用。

### 4. 空闲帧过滤

```python
idle_key = f"{recording_folderpath}--{file_path}"
keep_ranges = all_keep_ranges[idle_key]
```

空闲帧过滤是 DROID 数据处理的关键步骤。原始 DROID 数据中包含大量机器人静止不动的帧，这些帧对训练无益。脚本使用预计算的 `keep_ranges.json` 文件（来自 Physical Intelligence 的 OpenPI 项目），该文件记录了每个 episode 中非空闲帧的范围。

过滤过程：
```python
for key in episode_dict:
    tensor_parts = []
    for start_ix, end_ix in keep_ranges:
        tensor_parts.append(episode_dict[key][start_ix:end_ix])
    episode_dict[key] = np.concatenate(tensor_parts, axis=0)
```

仅保留 `keep_ranges` 指定的帧范围，丢弃空闲帧。

### 5. 视频编码

```python
def encode_video(frames: np.ndarray, output_path: Path, fps: int) -> None:
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("h264", rate=fps, options=options)
    # ...
```

使用 PyAV（FFmpeg 的 Python 封装）将图像帧序列编码为 H.264 MP4 视频。关键编码参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `preset` | `ultrafast` | 最快编码速度（牺牲压缩率） |
| `tune` | `zerolatency` | 零延迟模式 |
| `crf` | `23` | 恒定质量因子（越低质量越高） |
| `pix_fmt` | `yuv420p` | 标准像素格式 |

### 6. 两遍处理流程

`convert_droid_dataset()` 采用两遍处理：

**第一遍**：收集任务和过滤信息
```python
for i, episode in enumerate(tqdm.tqdm(dataset)):
    # 检查 episode 是否失败
    # 检查语言标注
    # 检查空闲帧
    if not filtered:
        kept_registry[i] = kept_count
        kept_count += 1
```

遍历所有 episode，收集唯一的语言任务描述，确定哪些 episode 需要保留，建立 `kept_registry` 映射。

**第二遍**：并行转换数据
```python
with ProcessPoolExecutor(mp_context=ctx, max_workers=max_workers) as executor:
    futures = [executor.submit(process_tfrecord, *args) for args in process_args]
```

使用 `spawn` 模式的多进程池并行处理各 TFRecord 分片，每个分片独立处理其中的 episode 并写入 Parquet 和 MP4 文件。

### 7. 元数据生成

转换完成后生成以下元数据：

| 文件 | 内容 |
|------|------|
| `meta/modality.json` | 状态/动作/视频/标注的字段映射 |
| `meta/tasks.jsonl` | 所有唯一任务描述及其索引 |
| `meta/episodes.jsonl` | 每个 episode 的索引、长度和关联任务 |
| `meta/info.json` | 数据集总体信息（episode 数、帧数、特征定义等） |

## 核心类/函数表格

| 函数 | 功能 |
|------|------|
| `convert_droid_dataset()` | 主转换函数，协调整个两遍处理流程 |
| `process_tfrecord()` | 处理单个 TFRecord 分片中的所有 episode |
| `process_sample()` | 处理单个 episode：拼接数据、过滤空闲帧、写入文件 |
| `_broadcast_metadata_rlds()` | 将轨迹元数据广播到每一帧 |
| `concat_state_or_action()` | 拼接状态/动作子字段并记录 concat_info |
| `encode_video()` | 将帧序列编码为 H.264 MP4 视频 |
| `tf_to_torch()` | TensorFlow 张量转 PyTorch 张量 |
| `tf_img_convert()` | 解码 TensorFlow 图像数据 |

## 与其他模块的关系

- **训练脚本**：转换产出的数据被 `scripts/train/droid_training_lora.sh` 和 `droid_training_full_finetune.sh` 使用
- **数据加载**：转换后的 LeRobot 格式数据由 `groot/vla/data/dataset/lerobot.py` 加载
- **预转换数据**：也可直接下载已转换的数据 `GEAR-Dreams/DreamZero-DROID-Data`，跳过此转换步骤
- **通用转换**：对于非 DROID 数据集，可使用 `convert_lerobot_to_gear.py` 进行元数据生成

## 使用示例

```bash
# 1. 下载原始 DROID 数据集
gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1 ./data/droid/1.0.1

# 2. 下载空闲帧过滤范围文件
gsutil cp gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json ./data/keep_ranges.json

# 3. 运行转换
python scripts/data/convert_droid.py ./data/droid/1.0.1 ./data/droid_lerobot \
    --keep-ranges-path ./data/keep_ranges.json --filter-failed

# 4. 仅转换前 5 个分片（调试用）
python scripts/data/convert_droid.py ./data/droid/1.0.1 ./data/droid_lerobot \
    --keep-ranges-path ./data/keep_ranges.json --first-n 5
```

## 总结

`convert_droid.py` 是一个功能完整的数据转换管线，将 DROID 1.0.1 的 RLDS 格式数据转换为 LeRobot v2 格式。脚本的核心特色包括：基于 OpenPI 的空闲帧过滤、多进程并行处理、PyAV 视频编码，以及完整的元数据生成。两遍处理策略先过滤后转换，确保输出数据的质量。脚本支持 `--first-n` 参数进行小规模调试，适合在正式转换前验证流程。
