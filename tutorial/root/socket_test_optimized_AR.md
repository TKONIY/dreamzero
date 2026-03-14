# `socket_test_optimized_AR.py` — DreamZero 分布式推理服务端

## 文件概述

`socket_test_optimized_AR.py` 是 DreamZero 项目的**核心推理服务端入口**。它负责:

1. 初始化多 GPU 分布式环境（通过 `torchrun` 启动）
2. 加载 DreamZero 视觉-语言-动作（VLA）模型
3. 通过 WebSocket 协议对外提供实时策略推理服务
4. 将来自 roboarena 格式的观测数据转换为模型内部的 AR_droid 格式
5. 在推理过程中自动录制和保存预测视频

该文件是整个项目从"训练好的模型"走向"实际机器人部署"的桥梁。

## 关键代码解析

### 1. 命令行参数定义

```python
@dataclasses.dataclass
class Args:
    port: int = 8000
    timeout_seconds: int = 50000  # 10 hours default, configurable
    model_path: str = "./checkpoints/dreamzero"
    enable_dit_cache: bool = False
    index: int = 0
    max_chunk_size: int | None = None
```

使用 `tyro` 库将 dataclass 自动转换为命令行参数。`timeout_seconds` 设定了分布式信号组的超时时间（默认约 14 小时），避免长时间推理时进程通信超时。`enable_dit_cache` 控制是否启用 DiT（Diffusion Transformer）缓存加速。

### 2. ARDroidRoboarenaPolicy — 格式转换适配器

这是本文件的核心类，起到**适配器（Adapter）**的作用，在 roboarena 评估框架和 DreamZero 模型之间转换数据格式。

#### 2.1 帧累积机制

```python
FRAMES_PER_CHUNK = 4

self._frame_buffers: dict[str, list[np.ndarray]] = {
    "video.exterior_image_1_left": [],
    "video.exterior_image_2_left": [],
    "video.wrist_image_left": [],
}
```

roboarena 每次发送单帧图像，但 DreamZero 模型采用自回归方式，需要多帧视频输入。此类通过维护三个摄像头的帧缓冲区来累积帧数据：
- 首次调用：使用 1 帧
- 后续调用：使用最近 4 帧（不足时用首帧填充）

#### 2.2 观测格式转换

```python
image_key_mapping = {
    "observation/exterior_image_0_left": "video.exterior_image_1_left",
    "observation/exterior_image_1_left": "video.exterior_image_2_left",
    "observation/wrist_image_left": "video.wrist_image_left",
}
```

roboarena 使用 0 起始索引命名摄像头（`exterior_image_0_left`），而 AR_droid 内部使用 1 起始索引（`exterior_image_1_left`）。同时还将状态数据从 1D 数组（`(7,)`）reshape 为 2D 数组（`(1, 7)`），并将 `prompt` 字段映射为 `annotation.language.action_text`。

#### 2.3 动作格式转换

```python
def _convert_action(self, action_dict: dict) -> np.ndarray:
    # joint_action: (N, 7), gripper_action: (N, 1)
    # 拼接为 (N, 8)
    action = np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32)
    return action
```

模型输出包含分离的关节位置和夹爪位置，此方法将它们拼接为 roboarena 要求的 `(N, 8)` 格式（7 个关节 + 1 个夹爪）。

#### 2.4 分布式推理流程

```python
def infer(self, obs: dict) -> np.ndarray:
    converted_obs = self._convert_observation(obs)

    # 通知 worker 进程继续（信号 0 = 继续）
    signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
    dist.broadcast(signal_tensor, src=0, group=self._signal_group)

    # 广播观测数据到所有 worker
    self._broadcast_batch_to_workers(converted_obs)

    # 所有 rank 同步执行前向传播
    dist.barrier()
    with torch.no_grad():
        result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
    dist.barrier()
```

推理过程跨多 GPU 协同完成：rank 0 通过 gloo 后端广播信号和数据，所有 rank 同步执行 `lazy_joint_forward_causal`（自回归因果前向传播），然后 rank 0 收集结果返回给客户端。

### 3. WebsocketPolicyServer — WebSocket 服务框架

```python
async def run(self, rank: int = 0):
    if rank == 0:
        async with _server.serve(self._handler, self._host, self._port, ...) as server:
            await server.serve_forever()
    else:
        await self._worker_loop()
```

rank 0 运行 WebSocket 服务器处理客户端连接，其他 rank 运行 `_worker_loop` 等待信号。信号协议如下：
- `0`：继续推理（接收数据并执行前向传播）
- `1`：关闭进程
- `2`：空闲等待（客户端断开后等待下一个连接）

### 4. 视频保存逻辑

```python
if len(self.video_across_time) > 10:
    video_across_time_cat = torch.cat(self.video_across_time, dim=2)
    frames = self._policy.trained_model.action_head.vae.decode(
        video_across_time_cat, tiled=..., tile_size=..., tile_stride=...
    )
    frames = rearrange(frames, "B C T H W -> B T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
```

每累积超过 10 个视频片段后，使用 VAE 解码器将潜在空间表示解码为像素帧，从 `[-1, 1]` 范围映射到 `[0, 255]` 的 uint8 格式，然后保存为 MP4 文件。文件名包含时间戳和帧数信息。

### 5. 分布式初始化

```python
def init_mesh() -> DeviceMesh:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size, ),
        mesh_dim_names=("ip", ),
    )
    return mesh
```

使用 NCCL 后端初始化进程组，创建一维设备网格（mesh），维度名为 `"ip"`（推理并行）。每个进程绑定到对应编号的 GPU。

### 6. main 函数 — 启动入口

```python
def main(args: Args) -> None:
    os.environ["ATTENTION_BACKEND"] = "TE"
    torch._dynamo.config.recompile_limit = 800

    device_mesh = init_mesh()
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        device="cuda",
        device_mesh=device_mesh,
    )
```

关键配置：
- 使用 TE（Transformer Engine）作为注意力后端
- 将 `torch._dynamo` 的重编译限制提高到 800，因为自回归推理会产生多种不同的张量形状
- 使用 gloo 后端创建信号通信组（gloo 支持 CPU 张量广播，NCCL 不支持）

## 核心类/函数

| 类/函数 | 用途 |
|---------|------|
| `Args` | 命令行参数 dataclass：端口、模型路径、缓存开关等 |
| `ARDroidRoboarenaPolicy` | 适配器类：在 roboarena 和 AR_droid 格式之间转换观测与动作 |
| `ARDroidRoboarenaPolicy._convert_observation()` | 将 roboarena 单帧观测转换为多帧 AR_droid 格式 |
| `ARDroidRoboarenaPolicy._convert_action()` | 将模型输出的关节/夹爪动作拼接为 `(N, 8)` 数组 |
| `ARDroidRoboarenaPolicy.infer()` | 执行一次完整推理：格式转换 -> 分布式广播 -> 前向传播 -> 动作提取 |
| `ARDroidRoboarenaPolicy._reset_state()` | 重置帧缓冲区并保存累积视频 |
| `WebsocketPolicyServer` | WebSocket 服务器：rank 0 处理连接，其他 rank 运行 worker 循环 |
| `WebsocketPolicyServer._worker_loop()` | 非 rank-0 进程的工作循环：等待信号、接收数据、参与前向传播 |
| `WebsocketPolicyServer._handler()` | WebSocket 连接处理器：接收观测、分发推理、返回动作 |
| `init_mesh()` | 初始化 NCCL 分布式环境和设备网格 |
| `_health_check()` | HTTP 健康检查端点（`/healthz`） |
| `main()` | 主入口：初始化环境、加载模型、启动服务 |

## 与其他模块的关系

- **`groot.vla.model.n1_5.sim_policy.GrootSimPolicy`**：核心策略模型，负责实际的视觉-语言-动作推理。本文件通过 `lazy_joint_forward_causal` 调用其自回归推理能力。
- **`groot.vla.data.schema.EmbodimentTag`**：定义具身智能体标签（如 `"oxe_droid"`），用于加载对应的模型配置。
- **`eval_utils.policy_server.WebsocketPolicyServer`**（即 `RoboarenaServer`）：roboarena 标准化的策略服务接口。rank 0 实际使用的是这个服务器，而非本文件中定义的 `WebsocketPolicyServer`。
- **`eval_utils.policy_server.PolicyServerConfig`**：服务器配置数据类，定义摄像头数量、分辨率、动作空间等参数。
- **`openpi_client`**：提供 msgpack 序列化和基础策略接口。
- **`tianshou.data.Batch`**：用作推理数据的容器结构。
- **`test_client_AR.py`**：本文件的客户端测试脚本。

## 总结

`socket_test_optimized_AR.py` 是 DreamZero 的**多 GPU 分布式推理服务端**。它将一个训练好的世界动作模型（World Action Model）部署为 WebSocket 服务，实现了从 roboarena 评估框架到 DreamZero 内部表示的完整数据流转换。核心设计包括：帧累积机制（将单帧流转化为多帧视频输入）、双通信后端架构（NCCL 用于张量计算同步、gloo 用于信号广播）、以及自动视频录制功能。该文件通过 `torchrun --nproc_per_node=8` 启动，支持 8 GPU 并行推理。
