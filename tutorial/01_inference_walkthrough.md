# DreamZero 推理流程：自顶向下执行详解

本文档从推理入口出发，按照代码实际执行顺序，逐层剖析 DreamZero 从接收一帧图像到输出机器人动作的完整链路。适合想要快速理解"代码到底怎么跑起来的"读者。

---

## 总览：推理执行调用链

```
torchrun socket_test_optimized_AR.py
  │
  ▼
main()                                          # 入口函数
  ├── init_mesh()                               # 初始化 NCCL 分布式
  ├── GrootSimPolicy(...)                       # 加载模型 + 变换
  │     ├── VLA.from_pretrained() / load_lora() # 加载权重
  │     ├── model.post_initialize()             # RoPE 频率 → GPU
  │     ├── model.parallelize(device_mesh)      # 张量并行
  │     └── eval_transform = instantiate(...)   # 数据变换链
  ├── ARDroidRoboarenaPolicy(groot_policy=...)  # 观测格式适配器
  └── RoboarenaServer(...).serve_forever()      # WebSocket 服务
        │
        ▼
  [客户端连接] → _handler() 接收 msgpack 观测
        │
        ▼
  ARDroidRoboarenaPolicy.infer(obs)
        ├── _convert_observation(obs)           # roboarena → DreamZero 格式
        ├── broadcast obs → 所有 GPU rank
        └── GrootSimPolicy.lazy_joint_forward_causal(batch)
              ├── apply(batch)                  # 数据变换(裁剪/resize/归一化)
              ├── VLA.lazy_joint_video_action_causal(inputs)
              │     ├── IdentityBackbone(inputs) → 透传
              │     └── WanFlowMatchingActionTF.lazy_joint_video_action(...)
              │           ├── encode_prompt()     # UMT5 文本编码
              │           ├── encode_image()      # CLIP ViT-H 图像编码
              │           ├── vae.encode()        # 视频 → 潜空间
              │           ├── DiT 去噪循环         # 视频+动作联合去噪
              │           │     └── WanModel.forward() × N steps
              │           ├── 提取 action_pred     # 动作预测结果
              │           └── 返回 video_pred      # 视频潜空间预测
              └── unapply(batch)                # 反归一化 → 绝对动作
        │
        ▼
  _convert_action(action_dict) → numpy (N, 8)
  → websocket.send(action) → 客户端
```

---

## 第 1 层：入口函数 `main()`

**文件**: [`socket_test_optimized_AR.py`](root/socket_test_optimized_AR.md) — 第 740 行

程序通过 `torchrun --nproc_per_node=8 socket_test_optimized_AR.py` 启动，每个 GPU 一个进程。

```python
def main(args: Args) -> None:
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ["ATTENTION_BACKEND"] = "TE"
    torch._dynamo.config.recompile_limit = 800

    device_mesh = init_mesh()       # ① 初始化 NCCL 分布式
    rank = dist.get_rank()

    signal_group = dist.new_group(backend="gloo", ...)  # ② gloo 信号组

    policy = GrootSimPolicy(        # ③ 加载模型
        embodiment_tag=EmbodimentTag("oxe_droid"),
        model_path=args.model_path,
        device="cuda",
        device_mesh=device_mesh,
    )

    wrapper_policy = ARDroidRoboarenaPolicy(groot_policy=policy, ...)  # ④ 格式适配器

    if rank == 0:
        RoboarenaServer(policy=wrapper_policy, ...).serve_forever()    # ⑤ Rank 0 启动 WebSocket
    else:
        WebsocketPolicyServer(...).run(server._worker_loop())          # ⑥ 其他 rank 进入 worker 循环
```

### 关键步骤解读

| 步骤 | 作用 | 涉及文件 |
|------|------|----------|
| ① `init_mesh()` | 初始化 NCCL 进程组和 DeviceMesh，为张量并行做准备 | `socket_test_optimized_AR.py:714` |
| ② 创建 gloo 信号组 | NCCL 不支持 CPU tensor 广播，用 gloo 后端做控制信号（continue/shutdown/idle） | `socket_test_optimized_AR.py:763` |
| ③ `GrootSimPolicy` | 加载模型权重、构建数据变换、设置推理参数（详见第 2 层） | [`sim_policy.py`](groot/vla/model/n1_5/sim_policy.md) |
| ④ `ARDroidRoboarenaPolicy` | 观测格式转换适配器，将 roboarena 格式转为 DreamZero 内部格式 | `socket_test_optimized_AR.py:44` |
| ⑤ `RoboarenaServer` | Rank 0 运行 WebSocket 服务，接收客户端请求 | [`policy_server.py`](eval_utils/policy_server.md) |
| ⑥ Worker 循环 | 非 Rank 0 进程等待信号，参与分布式前向传播 | `socket_test_optimized_AR.py:477` |

---

## 第 2 层：模型加载 — `GrootSimPolicy.__init__()`

**文件**: [`groot/vla/model/n1_5/sim_policy.py`](groot/vla/model/n1_5/sim_policy.md) — 第 205 行

这是推理侧最关键的初始化过程，完成三件事：**加载模型**、**构建数据变换**、**配置推理参数**。

### 2.1 加载模型

```python
# 从 checkpoint 目录的 conf.yaml 读取训练配置
train_cfg = OmegaConf.load(model_dir / "experiment_cfg" / "conf.yaml")

# 根据训练时是否使用 LoRA 选择加载方式
if train_cfg.save_lora_only:
    model = VLA.load_lora(model_path)       # 加载 LoRA 权重
else:
    model = VLA.from_pretrained(model_path) # 加载完整权重

# LoRA 训练的模型：合并 LoRA 权重到主模型（推理不需要 adapter 开销）
if model.action_head.train_architecture == "lora":
    model.action_head.model = model.action_head.model.merge_and_unload()

model.post_initialize()                     # RoPE 频率 → GPU
model.parallelize(device_mesh)              # 张量并行
```

模型加载链进入 [`VLA.from_pretrained()`](groot/vla/model/dreamzero/base_vla.md)，它会：

1. 读取 `config.json` → 创建 `VLAConfig`
2. 通过 `Hydra instantiate` 创建 backbone（IdentityBackbone）和 action_head（WanFlowMatchingActionTF）
3. 从 safetensors 文件加载权重

### 2.2 构建数据变换

```python
# 加载归一化统计信息
metadata = DatasetMetadata.model_validate(metadatas[embodiment_tag])

# 实例化变换链
eval_transform = instantiate(train_cfg.transforms[embodiment_tag])
eval_transform.set_metadata(metadata)
eval_transform.eval()  # 评估模式（关闭数据增强）
```

变换链 [`ComposedModalityTransform`](groot/vla/data/transform/base.md) 包含：
- [`VideoCrop`](groot/vla/data/transform/video.md) — 中心裁剪
- [`VideoResize`](groot/vla/data/transform/video.md) — 缩放到模型输入尺寸
- [`StateActionTransform`](groot/vla/data/transform/state_action.md) — 状态/动作归一化
- [`LanguageTransform`](groot/vla/data/transform/language.md) — 文本分词

### 2.3 配置推理参数

```python
self._video_delta_indices = np.array(modality_configs.video.eval_delta_indices)
# 例如 [-23, -16, -8, 0]，表示从历史帧中选取 4 帧作为输入
```

---

## 第 3 层：推理请求处理 — `ARDroidRoboarenaPolicy.infer()`

**文件**: `socket_test_optimized_AR.py` — 第 245 行

当客户端发送一帧观测时，执行流程如下：

```python
def infer(self, obs: dict) -> np.ndarray:
    # ① 会话管理：检测新会话 → 重置帧缓冲区
    if session_id != self._current_session_id:
        self._reset_state()

    # ② 格式转换：roboarena 观测 → DreamZero 内部格式
    converted_obs = self._convert_observation(obs)
    # 输入: {"observation/exterior_image_0_left": (H,W,3), ...}
    # 输出: {"video.exterior_image_1_left": (T,H,W,3), "state.joint_position": (1,7), ...}

    # ③ 信号同步：通知所有 GPU 准备推理
    dist.broadcast(signal_tensor, src=0, group=self._signal_group)  # gloo
    self._broadcast_batch_to_workers(converted_obs)                  # pickle → NCCL

    # ④ 核心推理（所有 rank 同步执行）
    dist.barrier()
    result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
    dist.barrier()

    # ⑤ 动作格式转换：DreamZero → numpy (N, 8)
    action = self._convert_action(action_chunk_dict)
    return action
```

### 帧累积机制

`_convert_observation()` 维护帧缓冲区，处理单帧→多帧的转换：

- **首次调用**：使用 1 帧（初始化）
- **后续调用**：累积到 `FRAMES_PER_CHUNK=4` 帧
- 每个摄像头独立缓冲，取最近 N 帧（不足时重复首帧填充）

---

## 第 4 层：数据变换与模型推理 — `lazy_joint_forward_causal()`

**文件**: [`groot/vla/model/n1_5/sim_policy.py`](groot/vla/model/n1_5/sim_policy.md) — 第 641 行

```python
def lazy_joint_forward_causal(self, batch, ...):
    # ① 保存原始观测（用于相对动作→绝对动作转换）
    original_obs = deep_copy(batch.obs)

    # ② 数据变换（ComposedModalityTransform）
    batch = self.apply(batch)  # → eval_transform(obs)
    # 视频: uint8 (T,H,W,3) → float32 归一化
    # 状态: 原始值 → q01/q99 归一化
    # 文本: str → token ids

    # ③ VLA 模型推理
    model_pred = self.trained_model.lazy_joint_video_action_causal(normalized_input)
    normalized_action = model_pred["action_pred"]   # (B, action_horizon, action_dim)
    video_pred = model_pred["video_pred"]           # 潜空间视频预测

    # ④ 反变换：归一化动作 → 真实物理值
    batch = self.unapply(Batch(normalized_action=normalized_action), obs=original_obs)
    # 包括: 反归一化 + 相对动作 → 绝对动作（加上当前关节位置）

    return batch, video_pred
```

---

## 第 5 层：VLA 模型前向传播 — `VLA.lazy_joint_video_action_causal()`

**文件**: [`groot/vla/model/dreamzero/base_vla.py`](groot/vla/model/dreamzero/base_vla.md) — 第 180 行

```python
def lazy_joint_video_action_causal(self, inputs, latent_video=None):
    backbone_inputs, action_inputs = self.prepare_input(inputs)
    backbone_outputs = self.backbone(backbone_inputs)           # IdentityBackbone → 透传
    action_head_outputs = self.action_head.lazy_joint_video_action(
        backbone_outputs, action_inputs, latent_video=latent_video
    )
    return action_head_outputs
```

VLA 模型由两部分组成：
- **Backbone**: [`IdentityBackbone`](groot/vla/model/dreamzero/backbone/identity.md) — 直接透传输入（DreamZero 不使用额外 backbone）
- **Action Head**: [`WanFlowMatchingActionTF`](groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.md) — 核心推理逻辑

---

## 第 6 层：核心推理引擎 — `WanFlowMatchingActionTF.lazy_joint_video_action()`

**文件**: [`groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`](groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.md) — 第 929 行

这是整个系统最核心的方法，执行联合视频-动作生成的完整流程：

```
输入: 视频帧 + 文本指令 + 机器人状态
          │
    ┌─────┴──────────────────────────────────────┐
    │                                            │
    ▼                                            ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ 文本编码      │   │ 图像编码      │   │ VAE 编码      │
  │ UMT5-XXL     │   │ CLIP ViT-H   │   │ WanVideoVAE  │
  │ text → emb   │   │ img → feat   │   │ video → z    │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         │                  │                   │
         ▼                  ▼                   ▼
  ┌────────────────────────────────────────────────────┐
  │          DiT 去噪循环 (N 步, 默认 10 步)             │
  │                                                    │
  │   噪声 z_T ──┐                                     │
  │              ▼                                     │
  │   ┌─────── WanModel.forward() ◄── 文本 emb ──┐    │
  │   │   因果分块注意力                  图像 feat  │    │
  │   │   ┌──────────┐  ┌───────────┐             │    │
  │   │   │ 视频通道  │  │ 动作通道   │             │    │
  │   │   │ z_{t-1}  │  │ a_{t-1}   │             │    │
  │   │   └──────────┘  └───────────┘             │    │
  │   │         ↓              ↓                  │    │
  │   └──── 预测速度场 v ──► 更新 z_t, a_t ────────┘    │
  │                                                    │
  │   重复 N 步 → z_0 (去噪后潜空间), a_0 (去噪后动作)   │
  └────────────────────────────────────────────────────┘
         │                            │
         ▼                            ▼
  video_pred (潜空间)          action_pred (B, H, D)
  (可选 VAE 解码为视频帧)       → 机器人动作序列
```

### 6.1 文本编码

```python
text_inputs = self._prepare_text_inputs(data)           # 分词
prompt_embs = [self.encode_prompt(text, mask) ...]       # UMT5-XXL 编码
```

**涉及文件**: [`wan_video_text_encoder.py`](groot/vla/model/dreamzero/modules/wan_video_text_encoder.md)

### 6.2 图像编码

```python
clip_feas, ys, image = self.encode_image(image, ...)    # CLIP ViT-H/14
self.clip_feas = clip_feas     # 缓存 (首帧编码一次，后续复用)
self.ys = ys
```

**涉及文件**: [`wan_video_image_encoder.py`](groot/vla/model/dreamzero/modules/wan_video_image_encoder.md)

关键优化：`current_start_frame == 0` 时才编码图像，后续自回归步骤复用缓存。

### 6.3 视频 VAE 编码

```python
image = self.vae.encode(videos, tiled=..., tile_size=..., tile_stride=...)
```

**涉及文件**: [`wan_video_vae.py`](groot/vla/model/dreamzero/modules/wan_video_vae.md)

将 RGB 视频帧编码为潜空间表示。支持分块编码（tiled）以减少显存占用。

### 6.4 Flow Matching 去噪循环

```python
# 初始化噪声
noise_obs = self.generate_noise(...)     # 视频潜空间噪声
noise_action = self.generate_noise(...)  # 动作噪声

# N 步去噪 (默认 10 步)
for i, t in enumerate(self.scheduler.timesteps):
    # DiT 前向传播：预测速度场
    model_output = self.model(
        x=[noise_obs, noise_action],     # 联合输入
        t=t,                              # 时间步
        context=prompt_embs,              # 文本条件
        clip_fea=self.clip_feas,          # 图像条件
        y=self.ys,                        # 图像特征
        ...
    )

    # 调度器更新：x_t → x_{t-1}
    noise_obs = self.scheduler.step(model_output_obs, t, noise_obs)
    noise_action = self.scheduler.step(model_output_action, t, noise_action)
```

**涉及文件**:
- DiT 模型: [`wan_video_dit_action_casual_chunk.py`](groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.md) — 因果分块注意力机制
- 调度器: [`flow_match_scheduler.py`](groot/vla/model/dreamzero/modules/flow_match_scheduler.md) 或 [`flow_unipc_multistep_scheduler.py`](groot/vla/model/dreamzero/modules/flow_unipc_multistep_scheduler.md)
- 注意力: [`wan2_1_attention.py`](groot/vla/model/dreamzero/modules/wan2_1_attention.md)、[`attention.py`](groot/vla/model/dreamzero/modules/attention.md)

### 6.5 提取输出

```python
action_pred = noise_action                              # (B, action_horizon, action_dim)
video_pred = noise_obs                                  # 潜空间，用于自回归下一步
return BatchFeature(action_pred=action_pred, video_pred=video_pred)
```

---

## 第 7 层：分布式推理协调

推理涉及 8 个 GPU 协同工作：

```
                    客户端
                      │
                      ▼
┌─────────── Rank 0 (WebSocket 服务) ──────────┐
│  接收观测 → gloo 信号广播 → NCCL 数据广播     │
│              ↓                                │
│     ┌── barrier() ──────────────────────┐     │
│     │   lazy_joint_forward_causal()     │     │
│     │   (DiT 张量并行 across 8 GPUs)    │     │
│     └── barrier() ──────────────────────┘     │
│              ↓                                │
│  提取动作 → 发送给客户端                       │
└───────────────────────────────────────────────┘
         ↕ gloo          ↕ NCCL
┌─── Rank 1-7 (Worker 循环) ───┐
│  等待信号 → 接收数据          │
│  → 参与 barrier()            │
│  → 执行同一份前向传播         │
│  → 等待下一轮信号             │
└──────────────────────────────┘
```

**两种通信后端**:
- **gloo (CPU)**：控制信号（0=继续, 1=关闭, 2=空闲等待下个客户端）
- **NCCL (GPU)**：张量并行中的 GPU 间通信（attention、all-reduce 等）

---

## 推荐阅读顺序

基于以上执行流程，建议按以下顺序阅读源码：

| 顺序 | 文件 | 理解目标 |
|------|------|----------|
| 1 | [`socket_test_optimized_AR.py`](root/socket_test_optimized_AR.md) | 推理入口、分布式初始化、WebSocket 服务 |
| 2 | [`eval_utils/policy_server.py`](eval_utils/policy_server.md) | WebSocket 协议、观测/动作格式定义 |
| 3 | [`groot/vla/model/n1_5/sim_policy.py`](groot/vla/model/n1_5/sim_policy.md) | 模型加载、数据变换、推理编排 |
| 4 | [`groot/vla/model/dreamzero/base_vla.py`](groot/vla/model/dreamzero/base_vla.md) | VLA 架构（backbone + action_head） |
| 5 | [`groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`](groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.md) | 核心推理：编码器 + Flow Matching 去噪 |
| 6 | [`groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`](groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.md) | DiT 因果分块注意力（最底层计算） |
| 7 | [`groot/vla/data/transform/`](groot/vla/data/transform/index.md) | 数据预处理与反变换 |
| 8 | [`groot/vla/configs/`](groot/vla/configs/index.md) | 理解所有超参数配置 |

---

## 关键概念速查

| 概念 | 含义 | 文件 |
|------|------|------|
| **自回归推理** | 每次预测一个 chunk 的视频+动作，将预测视频作为下一次的条件输入 | `wan_flow_matching_action_tf.py` |
| **因果分块注意力** | DiT 中的注意力机制，当前 chunk 只能看到历史 chunk，不能看到未来 | `wan_video_dit_action_casual_chunk.py` |
| **Flow Matching** | 替代 DDPM 的生成式方法，学习从噪声到数据的向量场，推理时 ODE 求解 | `flow_match_scheduler.py` |
| **LoRA 合并** | 推理时将 LoRA 适配器权重合并到主模型，消除 adapter 额外计算 | `sim_policy.py:300` |
| **DiT 缓存** | 缓存历史 chunk 的 KV，避免重复计算，通过 `ENABLE_DIT_CACHE` 开关控制 | `wan_video_dit_action_casual_chunk.py` |
| **相对动作** | 模型预测的是相对于当前状态的增量，需加上当前关节位置得到绝对值 | `sim_policy.py:unapply()` |
| **VRAM 管理** | 推理时按需加载/卸载编码器组件到 GPU，节省显存 | `vram_management.py` |

---

> 阅读建议：先通读本文建立全局执行流程的心智模型，再按推荐阅读顺序逐文件深入。遇到不理解的概念可回到 [背景知识](00_background_knowledge.md) 查阅。
