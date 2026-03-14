# `policy_client.py` — WebSocket 策略推理客户端

## 文件概述

`policy_client.py` 实现了一个基于 WebSocket 的策略推理客户端 `WebsocketClientPolicy`。它继承自 `openpi_client` 的 `BasePolicy` 抽象接口，将策略推理请求通过网络发送到远程服务器，并接收返回的动作结果。这种设计使得策略模型可以运行在 GPU 服务器上，而仿真环境或机器人控制程序运行在另一台机器上，通过网络通信实现解耦。

该文件改编自 [roboarena](https://github.com/robo-arena/roboarena/) 项目。

## 关键代码解析

### 1. Ping/Pong 超时配置

```python
PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600
```

WebSocket 协议默认每 20 秒发送一次 ping 帧，并期望在 20 秒内收到 pong 响应。但在策略推理场景中，服务器可能因为 GPU 推理耗时较长而无法及时回复。因此这里将 ping 间隔设为 60 秒，超时设为 600 秒（10 分钟），避免因推理延迟导致连接被意外关闭。

### 2. 客户端初始化与连接建立

```python
class WebsocketClientPolicy(BasePolicy):
    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()
```

初始化过程完成三件事：
- 构建 WebSocket URI（默认使用 `ws://` 协议）
- 创建 `msgpack_numpy.Packer` 实例用于序列化包含 numpy 数组的数据
- 调用 `_wait_for_server()` 建立连接并获取服务器元数据

### 3. 连接建立与协议回退

```python
def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
    logging.info(f"Waiting for server at {self._uri}...")
    try:
        conn = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            ping_interval=PING_INTERVAL_SECS,
            ping_timeout=PING_TIMEOUT_SECS,
        )
        metadata = msgpack_numpy.unpackb(conn.recv())
        return conn, metadata
    except:
        logging.info("Connection to server with ws:// failed. Trying wss:// ...")

    self._uri = "wss://" + self._uri.split("//")[1]
    conn = websockets.sync.client.connect(
        self._uri,
        compression=None,
        max_size=None,
        ping_interval=PING_INTERVAL_SECS,
        ping_timeout=PING_TIMEOUT_SECS,
    )
    metadata = msgpack_numpy.unpackb(conn.recv())
    return conn, metadata
```

这段代码实现了协议回退机制：
1. 首先尝试使用 `ws://`（非加密）连接服务器
2. 连接成功后，接收服务器发送的第一帧数据作为元数据（即 `PolicyServerConfig`）
3. 如果 `ws://` 连接失败，自动切换到 `wss://`（TLS 加密）重试

关键参数说明：
- `compression=None`：禁用压缩，减少 CPU 开销（图像数据压缩收益有限）
- `max_size=None`：不限制消息大小，因为图像观测数据可能很大

### 4. 推理请求

```python
@override
def infer(self, obs: Dict) -> Dict:
    obs["endpoint"] = "infer"
    data = self._packer.pack(obs)
    self._ws.send(data)
    response = self._ws.recv()
    if isinstance(response, str):
        raise RuntimeError(f"Error in inference server:\n{response}")
    return msgpack_numpy.unpackb(response)
```

推理流程：
1. 在观测字典中注入 `endpoint` 字段标记为 `"infer"`，用于服务器端路由
2. 使用 msgpack 序列化观测数据（支持 numpy 数组）
3. 发送数据并等待响应
4. **错误检测**：如果收到字符串类型的响应（而非二进制），说明服务器返回了错误信息的 traceback
5. 正常情况下反序列化二进制响应并返回动作字典

### 5. 重置请求

```python
@override
def reset(self, reset_info: Dict) -> None:
    reset_info["endpoint"] = "reset"
    data = self._packer.pack(reset_info)
    self._ws.send(data)
    response = self._ws.recv()
    return response
```

重置流程与推理类似，但通过 `endpoint` 字段标记为 `"reset"`。服务器收到后会调用策略模型的 `reset()` 方法，用于清除内部状态（如历史观测缓存）。

## 核心类/函数

| 类/函数 | 用途 |
|---------|------|
| `WebsocketClientPolicy` | WebSocket 策略客户端，实现 `BasePolicy` 接口，将推理请求转发到远程服务器 |
| `__init__(host, port)` | 初始化客户端，建立 WebSocket 连接 |
| `get_server_metadata()` | 返回服务器在握手时发送的配置元数据（`PolicyServerConfig`） |
| `_wait_for_server()` | 建立 WebSocket 连接，支持 ws/wss 协议回退 |
| `infer(obs)` | 发送观测数据到服务器，接收并返回推理动作 |
| `reset(reset_info)` | 通知服务器重置策略状态 |
| `PING_INTERVAL_SECS` | WebSocket ping 帧发送间隔（60 秒） |
| `PING_TIMEOUT_SECS` | WebSocket pong 响应超时时间（600 秒） |

## 与其他模块的关系

- **继承自** `openpi_client.base_policy.BasePolicy`：遵循统一的策略接口（`infer` / `reset`），使得客户端可以无缝替换本地策略
- **使用** `openpi_client.msgpack_numpy`：进行包含 numpy 数组的数据序列化/反序列化
- **与** `policy_server.py` **配对**：客户端发送的数据格式与服务器端的 `_handler` 方法预期格式一致
- **被** `run_sim_eval.py` **中的 `DreamZeroJointPosClient` 使用**：作为底层通信组件

## 总结

`policy_client.py` 是 DreamZero 评估管道中的网络通信层。它将策略推理抽象为简单的 `infer()` 和 `reset()` 调用，内部通过 WebSocket + msgpack 协议与远程服务器通信。关键设计包括：延长 ping/pong 超时以适应长时间推理、ws/wss 协议自动回退、通过 `endpoint` 字段在单一连接上复用推理和重置两种请求。这种设计使得策略模型和仿真环境可以灵活地部署在不同硬件上。
