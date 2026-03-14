# `test_client_AR.py` — AR_droid 策略服务端测试客户端

## 文件概述

`test_client_AR.py` 是 DreamZero 推理服务端 (`socket_test_optimized_AR.py`) 的**端到端测试客户端**。它通过 WebSocket 协议连接到策略服务器，发送观测数据（真实视频帧或零值填充图像），接收动作预测结果，并验证返回数据的格式正确性。

该文件在项目中扮演两个角色：
1. **集成测试工具**：验证服务端部署是否正常工作
2. **使用示例**：展示如何编写客户端与 DreamZero 服务进行交互

## 关键代码解析

### 1. 帧调度常量

```python
RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = 24
```

这两个常量定义了自回归推理的帧采样策略，与 `debug_inference.py` 保持一致：
- `RELATIVE_OFFSETS`：相对于当前锚帧的偏移量。例如锚帧为第 23 帧时，采样帧为 `[0, 7, 15, 23]`
- `ACTION_HORIZON`：每个动作块覆盖 24 帧的时间范围，因此下一个锚帧 = 当前锚帧 + 24

### 2. 视频帧加载

```python
def load_all_frames(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0)
```

从 `debug_image/` 目录加载三个摄像头的 MP4 视频文件，将 BGR 格式转换为 RGB 格式，返回 `(N, H, W, 3)` 的 numpy 数组。摄像头映射关系：

| roboarena 键名 | 视频文件 |
|----------------|----------|
| `observation/exterior_image_0_left` | `exterior_image_1_left.mp4` |
| `observation/exterior_image_1_left` | `exterior_image_2_left.mp4` |
| `observation/wrist_image_left` | `wrist_image_left.mp4` |

### 3. 帧调度构建

```python
def build_frame_schedule(total_frames: int, num_chunks: int) -> list[list[int]]:
    chunks = []
    current_frame = 23  # 第一个锚帧
    for _ in range(num_chunks):
        indices = [max(current_frame + off, 0) for off in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            break
        chunks.append(indices)
        current_frame += ACTION_HORIZON
    return chunks
```

生成完整的帧索引调度表。以默认 15 个 chunk 为例：
- 初始帧：`[0]`（单帧）
- Chunk 0：`[0, 7, 15, 23]`
- Chunk 1：`[24, 31, 39, 47]`
- Chunk 2：`[48, 55, 63, 71]`
- ...以此类推，每次锚帧前进 24 帧

### 4. 观测数据构造

```python
def _make_obs_from_video(camera_frames, frame_indices, prompt, session_id):
    obs = {}
    for cam_key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]  # (T, H, W, 3)
        if len(frame_indices) == 1:
            selected = selected[0]  # (H, W, 3) 单帧
        obs[cam_key] = selected
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs
```

将视频帧和状态信息组装为 roboarena 格式的观测字典。注意：
- 首次调用发送单帧 `(H, W, 3)`，后续发送 4 帧 `(4, H, W, 3)`
- 关节和夹爪位置使用零值（测试场景无需真实状态）
- `session_id` 确保服务端正确跟踪会话状态

### 5. 零值观测模式

```python
def _make_zero_observation(server_config, prompt, session_id):
    obs = {}
    h, w = server_config.image_resolution  # (180, 320)
    for i in range(server_config.n_external_cameras):
        obs[f"observation/exterior_image_{i}_left"] = np.zeros((h, w, 3), dtype=np.uint8)
    if server_config.needs_wrist_camera:
        obs["observation/wrist_image_left"] = np.zeros((h, w, 3), dtype=np.uint8)
    ...
```

当使用 `--use-zero-images` 参数时，生成全零图像的观测数据。这是一种快速冒烟测试（smoke test），不需要准备真实视频数据即可验证服务端连通性和基本推理流程。

### 6. 主测试流程

```python
def test_ar_droid_policy_server(host, port, num_chunks, prompt, use_zero_images):
    client = WebsocketClientPolicy(host=host, port=port)

    # 验证服务器元数据
    metadata = client.get_server_metadata()
    server_config = policy_server.PolicyServerConfig(**metadata)
    assert server_config.n_external_cameras == 2
    assert server_config.needs_wrist_camera
    assert server_config.action_space == "joint_position"

    # 初始帧
    obs = _make_obs_from_video(camera_frames, [0], prompt, session_id)
    actions = client.infer(obs)

    # 后续 chunk
    for chunk_idx, frame_indices in enumerate(chunks):
        obs = _make_obs_from_video(camera_frames, frame_indices, prompt, session_id)
        actions = client.infer(obs)

    # 触发服务端保存视频
    client.reset({})
```

测试流程按以下步骤执行：
1. 连接服务器并验证配置（2 个外部摄像头、腕部摄像头、关节位置动作空间）
2. 发送初始单帧观测
3. 按帧调度表依次发送 4 帧块
4. 发送 reset 信号触发服务端保存预测视频

### 7. 动作验证

```python
def _log_action(actions: np.ndarray, dt: float) -> None:
    assert isinstance(actions, np.ndarray)
    assert actions.ndim == 2
    assert actions.shape[-1] == 8  # 7 关节 + 1 夹爪
```

严格验证返回动作的格式：必须是 2D numpy 数组，最后一维为 8（7 个关节位置 + 1 个夹爪位置）。

## 核心类/函数

| 函数 | 用途 |
|------|------|
| `load_all_frames(video_path)` | 从 MP4 文件加载所有帧，返回 `(N, H, W, 3)` RGB 数组 |
| `load_camera_frames()` | 加载三个摄像头的所有视频帧 |
| `build_frame_schedule(total_frames, num_chunks)` | 构建帧索引调度表，每 chunk 包含 4 个帧索引 |
| `_make_obs_from_video(...)` | 从真实视频帧构建 roboarena 格式观测字典 |
| `_make_zero_observation(...)` | 构建全零图像的观测字典（冒烟测试用） |
| `test_ar_droid_policy_server(...)` | 核心测试函数：连接、验证、发送、接收、重置 |
| `_log_action(actions, dt)` | 验证并打印动作形状、值域和推理耗时 |
| `main()` | 命令行入口，解析参数并运行测试 |

## 与其他模块的关系

- **`socket_test_optimized_AR.py`**：本文件的测试对象，即推理服务端。通过 WebSocket 协议与之通信。
- **`eval_utils.policy_client.WebsocketClientPolicy`**：WebSocket 客户端实现类，封装了连接管理、数据序列化（msgpack）和请求/响应逻辑。
- **`eval_utils.policy_server.PolicyServerConfig`**：服务器配置数据类，客户端通过服务器返回的元数据构造此对象以验证配置正确性。
- **`debug_image/` 目录**：存放测试视频文件（三个摄像头视角），帧调度策略与 `debug_inference.py` 一致。

## 总结

`test_client_AR.py` 是一个功能完整的策略服务端测试工具，支持两种测试模式：使用真实视频帧的完整测试和使用零值图像的快速连通性测试。它精确复现了 DreamZero 自回归推理的帧调度策略（首帧单帧、后续每 24 帧采样 4 帧），并对返回的动作数据进行严格格式验证。通过 `session_id` 机制支持会话追踪，通过 `reset` 信号触发服务端的视频保存功能。
