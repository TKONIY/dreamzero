# video_utils.py - 视频处理工具

## 文件概述

`video_utils.py` 提供了多后端的视频帧提取工具，支持按帧索引或时间戳从视频文件中提取特定帧。该模块的核心设计是**后端无关**——同一个 API 可以使用不同的视频解码后端（ffmpeg、decord、torchcodec、opencv、pyav），通过 `video_backend` 参数切换。

这种设计使得项目可以根据部署环境灵活选择解码后端（例如某些 HPC 环境不支持 decord，可以回退到 ffmpeg）。

## 关键代码解析

### 1. FFmpeg 后端实现

模块内部实现了完整的 FFmpeg 命令行调用后端：

```python
def _get_video_info_ffmpeg(video_path: str) -> dict:  # 获取视频元数据
def _extract_frames_ffmpeg(video_path: str, frame_indices: list[int]) -> np.ndarray:  # 按索引提取
def _extract_frames_at_timestamps_ffmpeg(video_path: str, timestamps: list[float]) -> np.ndarray:  # 按时间戳提取
def _extract_all_frames_ffmpeg(video_path: str) -> tuple[np.ndarray, np.ndarray]:  # 提取全部帧
```

使用 `ffprobe` 获取视频信息（帧率、帧数、时长），使用 `ffmpeg` 以 rawvideo 格式输出帧数据，然后用 NumPy 解析。对于提取失败的帧会回退到黑帧。

### 2. 按帧索引提取 `get_frames_by_indices`

```python
def get_frames_by_indices(video_path, indices, video_backend="ffmpeg", video_backend_kwargs={}):
```

支持四种后端：
- **decord**：最高效的随机访问后端，`vr.get_batch(indices)` 一次获取多帧
- **torchcodec**：PyTorch 原生视频解码
- **ffmpeg**：通过命令行逐帧提取
- **opencv**：通过 `cv2.VideoCapture` 逐帧定位

### 3. 按时间戳提取 `get_frames_by_timestamps`

```python
def get_frames_by_timestamps(video_path, timestamps, video_backend="ffmpeg", fps=None):
```

这个函数更加复杂，因为需要处理时间戳到帧索引的映射：

- **decord**：获取所有帧时间戳 `get_frame_timestamp()`，然后用 `argmin` 找最近帧
- **torchcodec**：使用 `get_frames_played_at(seconds=timestamps)` 直接按时间提取。代码中包含了精度修正逻辑——将时间戳对齐到帧间隔的整数倍，避免浮点误差导致取到错误的帧
- **opencv**：类似 decord 的策略，先计算所有帧时间戳再 argmin
- **torchvision_av**：使用 PyAV 后端，通过 `seek` 定位关键帧后顺序解码

torchcodec 后端的时间戳修正逻辑值得注意：
```python
closest_timestamps = np.round(timestamps / interval) * interval
timestamps = np.clip(closest_timestamps, min_pts, max_pts)
```
这避免了类似 `1.39999998` 被误识别为上一帧的问题。

### 4. 提取全部帧 `get_all_frames`

```python
def get_all_frames(video_path, video_backend="ffmpeg") -> tuple[np.ndarray, np.ndarray]:
```

返回视频的所有帧和对应的时间戳数组。支持 decord、torchcodec、ffmpeg、pyav 四种后端。

## 核心类/函数表格

| 函数 | 说明 |
|------|------|
| `get_frames_by_indices(video_path, indices)` | 按帧索引提取指定帧 |
| `get_frames_by_timestamps(video_path, timestamps)` | 按时间戳提取指定帧 |
| `get_all_frames(video_path)` | 提取视频全部帧和时间戳 |
| `_get_video_info_ffmpeg(video_path)` | 使用 ffprobe 获取视频元数据 |
| `_extract_frames_ffmpeg(video_path, indices)` | FFmpeg 后端按索引提取 |
| `_extract_frames_at_timestamps_ffmpeg(video_path, ts)` | FFmpeg 后端按时间戳提取 |
| `_extract_all_frames_ffmpeg(video_path)` | FFmpeg 后端提取全部帧 |

**支持的后端**：

| 后端 | 依赖 | 特点 |
|------|------|------|
| `"ffmpeg"` | ffmpeg CLI | 无需 Python 包，兼容性最好 |
| `"decord"` | decord | 高效随机访问，GPU 解码支持 |
| `"torchcodec"` | torchcodec | PyTorch 原生，精度修正 |
| `"opencv"` | cv2 | 标准视觉库，逐帧定位 |
| `"pyav"` | av | PyAV，仅用于 `get_all_frames` |
| `"torchvision_av"` | torchvision | torchvision 的 PyAV 后端 |

## 与其他模块的关系

- **外部依赖**：`cv2`、`numpy`、`av`、`torchvision`，以及可选的 `decord`、`torchcodec`
- **被数据加载模块使用**：从视频格式的演示数据中提取训练帧
- **独立性强**：不依赖项目内的其他工具模块

## 总结

`video_utils.py` 通过统一的 API 抽象了多种视频解码后端，让项目可以灵活适应不同的部署环境。FFmpeg 命令行后端作为保底方案确保了最大兼容性，而 decord 和 torchcodec 则提供了更好的性能。按时间戳提取帧的功能中包含的浮点精度修正逻辑，反映了在实际使用中遇到的工程问题和解决方案。所有函数的返回值统一为 NumPy 数组（形状 `[N, H, W, 3]`），简化了下游处理。
