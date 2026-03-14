# `pyproject.toml` — 项目构建与依赖配置

## 文件概述

`pyproject.toml` 是 DreamZero 项目的**构建系统配置文件**，遵循 PEP 621 标准。它定义了项目元数据、Python 版本约束、所有运行时依赖、开发依赖、命令行入口点以及包发现规则。这是安装和分发 DreamZero 的唯一配置入口。

## 关键代码解析

### 1. 构建系统

```toml
[build-system]
requires = ["setuptools>=67", "wheel", "pip"]
build-backend = "setuptools.build_meta"
```

使用 setuptools 作为构建后端，要求 setuptools 版本不低于 67。通过 `pip install -e .` 或 `pip install .` 即可安装项目。

### 2. 项目元数据

```toml
[project]
name = "dreamzero"
description = "DreamZero: World Action Models Are Zero-Shot Policies"
version = "1.0.0"
requires-python = "~=3.11,<3.13"
```

- 项目全名 `dreamzero`，版本 `1.0.0`
- **Python 版本要求**：`~=3.11,<3.13`，即支持 Python 3.11.x 和 3.12.x，不支持 3.10 及以下或 3.13 及以上
- 作者为 **NVIDIA Gear Lab**
- 采用 Apache 2.0 开源许可证

### 3. 核心依赖分类解析

DreamZero 的依赖可按功能分为以下几组：

#### 深度学习框架

```toml
"torch==2.8.0",
"torchvision==0.23.0",
"torchaudio==2.8.0",
"deepspeed",
"peft==0.5.0",
"transformers==4.51.3",
"diffusers==0.30.2",
```

- **PyTorch 2.8.0**：核心计算框架，版本锁定以确保兼容性
- **DeepSpeed**：分布式训练和推理优化
- **PEFT 0.5.0**：参数高效微调（LoRA 等）
- **Transformers**：Hugging Face 模型库
- **Diffusers**：扩散模型工具库，用于 flow matching 调度器等

#### 视频与图像处理

```toml
"av==15.0.0",
"opencv-python==4.8.0.74",
"imageio==2.34.2",
"imageio-ffmpeg",
"albumentations==1.4.18",
"decord2",
```

- **av / imageio / decord2**：视频读写与解码
- **OpenCV**：图像处理
- **albumentations**：数据增强

#### NVIDIA 加速

```toml
"nvidia-modelopt",
"nvidia-modelopt-core",
"tensorrt",
```

NVIDIA 模型优化和 TensorRT 推理引擎，用于生产环境的高性能推理。

#### 机器人与仿真

```toml
"gymnasium",
"mujoco",
"pin",
"pin-pink",
"pybullet; sys_platform == 'linux'",
"gear",
```

- **gymnasium / mujoco / pybullet**：机器人仿真环境
- **pin / pin-pink**：Pinocchio 运动学库
- **gear**：NVIDIA 内部机器人工具库

#### 通信与序列化

```toml
"flask",
"flask_socketio",
"python-socketio>=5.13.0",
"websockets" (通过 openpi-client 间接依赖),
"msgpack",
"msgpack-numpy",
"openpi-client==0.1.1",
"pyzmq",
"redis",
```

支持 WebSocket、Socket.IO、ZMQ、Redis 等多种通信协议，适应不同部署场景。

#### 数据与实验管理

```toml
"hydra-core",
"ray[default]==2.47.1",
"wandb",
"datasets==3.6.0",
"lmdb",
"multi-storage-client[boto3,msal,observability-otel]==0.33.0",
```

- **Hydra**：配置管理框架
- **Ray**：分布式计算与超参搜索
- **W&B**：实验追踪
- **datasets**：Hugging Face 数据集库
- **multi-storage-client**：多云存储客户端（支持 S3、Azure 等）

#### 工具库

```toml
"einops==0.8.1",
"tyro",
"tianshou==0.5.1",
"timm",
"scipy==1.15.3",
"numpy==1.26.4",
```

- **einops**：张量维度变换
- **tyro**：命令行参数自动生成
- **tianshou**：强化学习框架（提供 Batch 数据结构）
- **timm**：视觉模型库

### 4. 平台条件依赖

```toml
"PyQt6; platform_machine != 'aarch64'",
"evdev; sys_platform == 'linux'",
"pybullet; sys_platform == 'linux'",
```

部分依赖仅在特定平台安装：
- `PyQt6` 在 ARM64（如 Jetson）上不安装
- `evdev` 和 `pybullet` 仅在 Linux 上安装

### 5. 开发依赖

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
]
```

通过 `pip install -e ".[dev]"` 安装开发工具：pytest（测试）、black（格式化）、isort（import 排序）。

### 6. 命令行入口点

```toml
[project.scripts]
dreamzero-server = "socket_test_optimized_AR:main"
```

安装后可直接执行 `dreamzero-server` 命令启动推理服务，等同于运行 `socket_test_optimized_AR.py` 中的 `main` 函数。

### 7. 包发现规则

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["groot*"]
```

仅将 `groot` 及其子包打包分发。根目录的 `.py` 文件（如 `socket_test_optimized_AR.py`）不作为包的一部分，但通过 `[project.scripts]` 入口点仍可被调用。

## 核心配置项

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `name` | `dreamzero` | 包名 |
| `version` | `1.0.0` | 当前版本 |
| `requires-python` | `~=3.11,<3.13` | 支持 Python 3.11-3.12 |
| `build-backend` | `setuptools.build_meta` | 使用 setuptools 构建 |
| `dreamzero-server` | `socket_test_optimized_AR:main` | CLI 入口点 |
| `packages.include` | `groot*` | 仅打包 groot 包 |

## 与其他模块的关系

- **`socket_test_optimized_AR.py`**：通过 `[project.scripts]` 注册为 `dreamzero-server` 命令行工具。
- **`groot/` 包**：通过 `[tool.setuptools.packages.find]` 指定为项目的核心代码包。
- **`eval_utils/`**：未被显式包含在包发现规则中，作为辅助工具模块存在。

## 总结

`pyproject.toml` 定义了 DreamZero 项目完整的依赖生态。项目依赖超过 60 个包，覆盖深度学习（PyTorch 2.8.0 全家桶）、机器人仿真（MuJoCo、Pinocchio）、视频处理（av、OpenCV、imageio）、分布式计算（DeepSpeed、Ray）、NVIDIA 加速（TensorRT、ModelOpt）以及实验管理（W&B、Hydra）等领域。严格的版本锁定策略（大部分核心依赖指定精确版本）确保了环境的可复现性。
