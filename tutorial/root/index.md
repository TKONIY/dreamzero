# DreamZero 根目录模块概览

## 项目简介

DreamZero（"World Action Models Are Zero-Shot Policies"）是由 NVIDIA Gear Lab 开发的视觉-语言-动作（VLA）模型框架。该项目的核心思想是将世界模型（World Model）直接用作零样本策略（Zero-Shot Policy），通过视频生成式的自回归推理来预测机器人动作。

## 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DreamZero 项目架构                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐     WebSocket      ┌───────────────────┐  │
│  │  test_client_AR.py  │ ──────────────────> │ socket_test_      │  │
│  │  (测试客户端)        │ <────────────────── │ optimized_AR.py   │  │
│  │                     │   观测 / 动作        │ (推理服务端)       │  │
│  └─────────────────────┘                     └───────┬───────────┘  │
│          │                                           │              │
│          │ 使用                                       │ 使用         │
│          v                                           v              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                     eval_utils/                              │    │
│  │  ┌──────────────────────┐  ┌─────────────────────────────┐  │    │
│  │  │ policy_client.py     │  │ policy_server.py            │  │    │
│  │  │ WebsocketClientPolicy│  │ WebsocketPolicyServer       │  │    │
│  │  │ (客户端基类)          │  │ PolicyServerConfig          │  │    │
│  │  └──────────────────────┘  └─────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                    │                                │
│                                    │ 调用                           │
│                                    v                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                       groot/ (核心包)                        │    │
│  │                                                             │    │
│  │  ┌───────────────────────────────────────────────────────┐  │    │
│  │  │                   groot.vla.model/                     │  │    │
│  │  │                                                       │  │    │
│  │  │  ┌─────────────────┐    ┌──────────────────────────┐  │  │    │
│  │  │  │  n1_5/          │    │  dreamzero/               │  │  │    │
│  │  │  │  sim_policy.py  │───>│  base_vla.py             │  │  │    │
│  │  │  │  (推理策略)      │    │  action_head/            │  │  │    │
│  │  │  └─────────────────┘    │    wan_flow_matching_     │  │  │    │
│  │  │                         │    action_tf.py           │  │  │    │
│  │  │                         │  modules/                 │  │  │    │
│  │  │                         │    wan_video_dit*.py      │  │  │    │
│  │  │                         │    wan_video_vae.py       │  │  │    │
│  │  │                         │    flow_match_scheduler   │  │  │    │
│  │  │                         │    attention.py           │  │  │    │
│  │  │                         │  backbone/                │  │  │    │
│  │  │                         │  transform/               │  │  │    │
│  │  │                         └──────────────────────────┘  │  │    │
│  │  └───────────────────────────────────────────────────────┘  │    │
│  │                                                             │    │
│  │  ┌───────────────────────────────────────────────────────┐  │    │
│  │  │                   groot.vla.data/                      │  │    │
│  │  │  dataset/     数据集加载 (LeRobot, 分片)                │  │    │
│  │  │  schema/      数据格式定义 (EmbodimentTag)              │  │    │
│  │  │  transform/   数据变换 (视频, 状态, 语言)                │  │    │
│  │  │  conversion/  数据转换工具 (DROID, AgiBOT)              │  │    │
│  │  └───────────────────────────────────────────────────────┘  │    │
│  │                                                             │    │
│  │  ┌───────────────────────────────────────────────────────┐  │    │
│  │  │  groot.vla.experiment/  实验管理与训练循环               │  │    │
│  │  │  groot.vla.common/      通用工具 (IO, 数组, 图像)        │  │    │
│  │  │  groot.vla.utils/       动作参数覆盖, 计时器             │  │    │
│  │  └───────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─────────────────┐     ┌──────────────────────────────────────┐   │
│  │ pyproject.toml   │     │ scripts/                             │   │
│  │ (构建/依赖配置)   │     │   data/convert_*.py  数据转换脚本    │   │
│  │                  │     │   open_loop_yam.py   开环评估        │   │
│  └─────────────────┘     └──────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    外部依赖                                   │   │
│  │  torch.distributed (NCCL + gloo)  分布式通信                  │   │
│  │  openpi_client                    WebSocket + msgpack         │   │
│  │  tianshou.data.Batch              数据容器                    │   │
│  │  Wan Video DiT / VAE              视频生成模型                │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## 数据流

```
                        推理数据流
                        =========

  客户端                     服务端 (Rank 0)              Worker (Rank 1..N)
    │                            │                            │
    │  1. 发送观测 (roboarena)    │                            │
    │  ─────────────────────>    │                            │
    │                            │  2. 格式转换                │
    │                            │  roboarena -> AR_droid      │
    │                            │                            │
    │                            │  3. 广播信号 (gloo)         │
    │                            │  ─────────────────────>    │
    │                            │                            │
    │                            │  4. 广播观测数据 (pickle)    │
    │                            │  ─────────────────────>    │
    │                            │                            │
    │                            │  5. dist.barrier()         │
    │                            │  <────────────────────>    │
    │                            │                            │
    │                            │  6. 分布式前向传播           │
    │                            │  lazy_joint_forward_causal │
    │                            │  (NCCL 张量并行)            │
    │                            │  <────────────────────>    │
    │                            │                            │
    │                            │  7. dist.barrier()         │
    │                            │  <────────────────────>    │
    │                            │                            │
    │  8. 返回动作 (N, 8)         │                            │
    │  <─────────────────────    │                            │
    │                            │                            │
```

## 根目录文件索引

| 文件 | 类型 | 说明 | 详细文档 |
|------|------|------|----------|
| `socket_test_optimized_AR.py` | 服务端 | 多 GPU 分布式推理 WebSocket 服务，将 DreamZero 模型部署为实时策略服务 | [socket_test_optimized_AR.md](./socket_test_optimized_AR.md) |
| `test_client_AR.py` | 测试客户端 | 端到端测试工具，支持真实视频帧和零值图像两种测试模式 | [test_client_AR.md](./test_client_AR.md) |
| `pyproject.toml` | 配置文件 | 项目构建、60+ 依赖管理、CLI 入口点定义 | [pyproject.md](./pyproject.md) |

## 核心技术栈

- **模型架构**：基于 Wan Video DiT（Diffusion Transformer）的世界动作模型，使用 Flow Matching 作为生成调度策略，VAE 进行视频编解码
- **推理方式**：自回归因果推理（Autoregressive Causal Inference），首帧单帧输入，后续每步输入 4 帧视频
- **分布式方案**：PyTorch Distributed + DeviceMesh，NCCL 后端用于张量并行计算，gloo 后端用于 CPU 信号广播
- **通信协议**：WebSocket + msgpack-numpy 序列化，支持健康检查端点 (`/healthz`)
- **数据格式**：roboarena 标准接口（观测/动作），内部使用 AR_droid 格式（多帧视频 + 状态向量）

## 快速启动

```bash
# 安装项目
pip install -e .

# 启动 8 GPU 推理服务
torchrun --nproc_per_node=8 socket_test_optimized_AR.py \
    --port 8000 \
    --model-path ./checkpoints/dreamzero

# 运行测试客户端（使用真实视频帧）
python test_client_AR.py --host localhost --port 8000

# 运行快速连通性测试（零值图像）
python test_client_AR.py --host localhost --port 8000 --use-zero-images
```
