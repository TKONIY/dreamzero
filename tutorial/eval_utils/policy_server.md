# `policy_server.py` — WebSocket 策略推理服务器

## 文件概述

`policy_server.py` 实现了一个基于异步 WebSocket 的策略推理服务器。它将任意实现了 `BasePolicy` 接口的策略模型封装为网络服务，接受来自客户端的观测数据，执行推理后返回动作。服务器使用 `asyncio` 异步框架处理 WebSocket 连接，同时通过 `PolicyServerConfig` 数据类向客户端通告服务器的能力与配置需求。

该文件同样改编自 [roboarena](https://github.com/robo-arena/roboarena/) 项目。

## 关键代码解析

### 1. 服务器配置数据类

```python
@dataclasses.dataclass
class PolicyServerConfig:
    image_resolution: tuple[int, int] | None = (224, 224)
    needs_wrist_camera: bool = True
    n_external_cameras: int = 1
    needs_stereo_camera: bool = False
    needs_session_id: bool = False
    action_space: str = "joint_position"
```

`PolicyServerConfig` 定义了服务器对客户端的配置要求：

| 字段 | 默认值 | 含义 |
|------|--------|------|
| `image_resolution` | `(224, 224)` | 客户端发送图像前应缩放到的分辨率，`None` 表示不缩放 |
| `needs_wrist_camera` | `True` | 是否需要腕部相机图像 |
| `n_external_cameras` | `1` | 需要的外部相机数量（0/1/2） |
| `needs_stereo_camera` | `False` | 是否需要立体相机图像 |
| `needs_session_id` | `False` | 是否需要唯一的评估会话 ID（用于策略内部追踪历史） |
| `action_space` | `"joint_position"` | 动作空间类型：关节位置/速度 或 笛卡尔位置/速度 |

这个配置在连接建立时作为握手信息发送给客户端，使客户端知道应该发送哪些数据以及如何预处理。

### 2. 服务器类与观测/动作接口

```python
class WebsocketPolicyServer:
    """
    Interface:
      Observation:
        - observation/wrist_image_left: (H, W, 3)
        - observation/exterior_image_{i}_left: (H, W, 3)
        - observation/joint_position: (7,)
        - observation/cartesian_position: (6,)
        - observation/gripper_position: (1,)
        - prompt: str

      Action:
        - action: (N, 8) or (N, 7)
    """
```

文档字符串详细定义了观测和动作的数据格式，这是客户端和服务器之间的数据协议。动作维度的含义为：7 个关节动作（或 6 个笛卡尔动作）加 1 个夹爪位置。`N` 表示动作块（action chunk）的长度，即一次推理返回多步动作。

### 3. 服务器初始化

```python
def __init__(
    self,
    policy: BasePolicy,
    server_config: PolicyServerConfig,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    self._policy = policy
    self._server_config = server_config
    self._host = host
    self._port = port
    logging.getLogger("websockets.server").setLevel(logging.INFO)
```

服务器接受一个策略对象和配置对象。`host` 默认为 `"0.0.0.0"` 表示监听所有网络接口，`port` 默认为 8000。最后一行将 `websockets` 库的日志级别设为 `INFO`，方便观察连接事件。

### 4. 异步服务启动

```python
def serve_forever(self) -> None:
    asyncio.run(self.run())

async def run(self):
    async with websockets.asyncio.server.serve(
        self._handler,
        self._host,
        self._port,
        compression=None,
        max_size=None,
    ) as server:
        await server.serve_forever()
```

`serve_forever()` 是阻塞式的入口方法，内部通过 `asyncio.run()` 启动异步事件循环。`compression=None` 和 `max_size=None` 的配置与客户端保持一致——禁用压缩并允许任意大小的消息。

### 5. 连接处理核心逻辑

```python
async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
    logging.info(f"Connection from {websocket.remote_address} opened")
    packer = msgpack_numpy.Packer()

    # 发送服务器配置作为握手信息
    await websocket.send(packer.pack(dataclasses.asdict(self._server_config)))

    while True:
        try:
            obs = msgpack_numpy.unpackb(await websocket.recv())

            endpoint = obs["endpoint"]
            del obs["endpoint"]
            if endpoint == "reset":
                self._policy.reset(obs)
                to_return = "reset successful"
            else:
                action = self._policy.infer(obs)
                to_return = packer.pack(action)
            await websocket.send(to_return)
        except websockets.ConnectionClosed:
            logging.info(f"Connection from {websocket.remote_address} closed")
            break
        except Exception:
            await websocket.send(traceback.format_exc())
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="Internal server error. Traceback included in previous frame.",
            )
            raise
```

这是服务器最核心的代码，每个客户端连接都会创建一个 `_handler` 协程实例。其工作流程为：

**握手阶段：**
- 连接建立后立即发送 `PolicyServerConfig`（序列化为字典），让客户端知道服务器的配置要求

**请求处理循环：**
1. 接收并反序列化客户端发送的观测数据
2. 提取 `endpoint` 字段判断请求类型（`"infer"` 或 `"reset"`），然后从数据中删除该字段
3. 根据端点类型调用策略的 `infer()` 或 `reset()` 方法
4. 推理结果序列化为二进制返回；重置结果返回字符串 `"reset successful"`

**错误处理：**
- `ConnectionClosed`：客户端断开连接，正常退出循环
- 其他异常：将完整的 traceback 发送给客户端（便于远程调试），然后关闭连接并重新抛出异常

### 6. 测试用例

```python
if __name__ == "__main__":
    class DummyPolicy(BasePolicy):
        def infer(self, obs):
            return np.zeros((1, 8), dtype=np.float32)

        def reset(self, reset_info):
            pass

    policy = DummyPolicy()
    server = WebsocketPolicyServer(policy, PolicyServerConfig())
    server.serve_forever()
```

文件底部提供了一个简单的测试用例，使用 `DummyPolicy`（始终返回全零动作）启动服务器，可以配合 `policy_client.py` 进行端到端通信测试。

## 核心类/函数

| 类/函数 | 用途 |
|---------|------|
| `PolicyServerConfig` | 数据类，定义服务器对客户端的配置需求（图像分辨率、相机选择、动作空间等） |
| `WebsocketPolicyServer` | WebSocket 策略服务器，托管策略模型并提供推理服务 |
| `__init__(policy, server_config, host, port)` | 初始化服务器，绑定策略对象和配置 |
| `serve_forever()` | 阻塞式启动服务器，开始监听连接 |
| `run()` | 异步服务器主循环 |
| `_handler(websocket)` | 单个客户端连接的处理协程，包含握手、请求分发和错误处理 |

## 与其他模块的关系

- **依赖** `openpi_client.base_policy.BasePolicy`：策略模型必须实现该接口
- **使用** `openpi_client.msgpack_numpy`：与客户端共享相同的序列化方案
- **与** `policy_client.py` **配对**：服务器发送的握手数据和响应格式与客户端的预期一致
- **被外部脚本调用**：通常在独立终端中启动，例如 openpi 框架的 `serve_policy.py` 脚本

## 总结

`policy_server.py` 是 DreamZero 评估架构中的服务端组件。它使用异步 WebSocket 服务器托管策略模型，通过 `PolicyServerConfig` 实现客户端与服务器之间的配置协商，并通过 `endpoint` 字段在单一连接上复用推理和重置两种操作。异步架构使其能够高效处理连接管理，而详细的错误处理（将 traceback 发送给客户端）极大地方便了远程调试。整体设计使得策略模型可以独立于仿真或机器人环境运行，支持灵活的分布式部署。
