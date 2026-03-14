# eval_utils 模块概览

## 模块简介

`eval_utils` 是 DreamZero 项目中负责**策略评估**的核心模块。它实现了一套基于 WebSocket 的客户端-服务器架构，使得策略模型的推理与仿真环境的执行可以在不同进程（甚至不同机器）上独立运行，从而实现解耦和灵活部署。

## 架构总览

```
+------------------------------------------------------+
|                   run_sim_eval.py                     |
|   (仿真评估主脚本 - Isaac Lab 环境 + 评估循环)       |
|                                                       |
|   +------------------+     +----------------------+   |
|   | DreamZeroJoint   |     | Isaac Lab 仿真环境   |   |
|   | PosClient        |     | (DROID Gymnasium)    |   |
|   |                  |     |                      |   |
|   | - 观测提取       |     | - 场景管理           |   |
|   | - 动作分块执行   |     | - 物理仿真           |   |
|   | - 夹爪二值化     |     | - 视频录制           |   |
|   +--------+---------+     +----------+-----------+   |
|            |                          |               |
|            | infer(obs)               | step(action)  |
|            v                          ^               |
+------------------------------------------------------+
             |
             | WebSocket (msgpack)
             v
+------------------------------------------------------+
|                  policy_client.py                     |
|   (WebSocket 客户端 - 序列化/反序列化)               |
|                                                       |
|   WebsocketClientPolicy                              |
|   - 连接管理 (ws:// / wss://)                        |
|   - msgpack 编解码                                    |
|   - infer() / reset() 端点路由                       |
+------------------------------------------------------+
             |
             | WebSocket 通信
             | (二进制帧: msgpack + numpy)
             v
+------------------------------------------------------+
|                  policy_server.py                     |
|   (WebSocket 服务器 - 异步处理)                      |
|                                                       |
|   WebsocketPolicyServer                              |
|   - asyncio 事件循环                                 |
|   - 连接握手 (发送 PolicyServerConfig)               |
|   - 请求分发 (infer / reset)                         |
|   - 错误处理与连接关闭                               |
|                                                       |
|   PolicyServerConfig                                 |
|   - 图像分辨率、相机配置                             |
|   - 动作空间类型                                     |
+------------------------------------------------------+
             |
             | 调用
             v
+------------------------------------------------------+
|              BasePolicy (openpi_client)               |
|   (策略模型抽象接口)                                 |
|   - infer(obs) -> action                             |
|   - reset(reset_info)                                |
+------------------------------------------------------+
```

## 数据流

```
仿真环境 obs (图像+关节状态)
        |
        v
DreamZeroJointPosClient._extract_observation()
  -> 提取图像、关节位置、夹爪位置
        |
        v
image_tools.resize_with_pad() 缩放图像至 180x320
        |
        v
WebsocketClientPolicy.infer()
  -> msgpack 序列化 -> WebSocket 发送
        |
        v
WebsocketPolicyServer._handler()
  -> msgpack 反序列化 -> policy.infer(obs) -> 序列化返回
        |
        v
动作块 (N, 8) 返回至客户端
  -> 逐步执行，夹爪值二值化
        |
        v
env.step(action) 执行动作
```

## 文件清单

| 文件 | 用途 |
|------|------|
| `policy_client.py` | WebSocket 客户端，封装策略推理的网络通信 |
| `policy_server.py` | WebSocket 服务器，托管策略模型并提供推理服务 |
| `run_sim_eval.py` | 仿真评估主脚本，驱动 Isaac Lab 环境进行策略评估 |

## 外部依赖

| 依赖 | 用途 |
|------|------|
| `openpi_client` | 提供 `BasePolicy` 接口、`msgpack_numpy` 序列化、`image_tools` 图像工具 |
| `websockets` | WebSocket 协议实现（同步客户端 + 异步服务器） |
| `sim_evals` | Isaac Lab 仿真环境定义与抽象推理客户端接口 |
| `isaaclab` / `isaaclab_tasks` | NVIDIA Isaac Lab 仿真框架 |
| `tyro` | 命令行参数解析 |
| `mediapy` | 视频录制与保存 |
