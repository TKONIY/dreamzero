# wan_flow_matching_action_tf.yaml -- WAN Flow Matching 动作头配置

## 文件概述

`wan_flow_matching_action_tf.yaml` 定义了 DreamZero 的**核心动作头模块** -- WANPolicyHead。该模块基于 WAN（万相）视频扩散模型，通过 Flow Matching 方法同时生成视频预测和机器人动作序列。这是 DreamZero 区别于传统 VLA 模型的关键创新：将动作预测嵌入到视频生成的扩散过程中。

## 关键配置项解析

### 1. 顶层全局参数

```yaml
add_pos_embed: true
hidden_size: 64
attn_dropout: 0.2
repa_layer: 8
repa_coeff: 1.0
```

- `add_pos_embed`：是否添加位置编码。
- `hidden_size`：投影层的隐藏维度（64 维）。
- `attn_dropout`：注意力层 Dropout 比率。
- `repa_layer` / `repa_coeff`：表征对齐（Representation Alignment）的层数和系数，用于对齐视觉表征。

### 2. 训练架构与 LoRA

```yaml
train_architecture: "lora"
lora_rank: 4
lora_alpha: 4
lora_target_modules: "q,k,v,o,ffn.0,ffn.2"
init_lora_weights: "kaiming"
```

DreamZero 默认使用 **LoRA（Low-Rank Adaptation）** 微调策略，仅在 WAN 模型的 Q/K/V/O 注意力投影和 FFN 层插入低秩适配器（rank=4），大幅减少可训练参数量。

### 3. 视频分块处理（Tiled Processing）

```yaml
tiled: false
tile_size_height: 34
tile_size_width: 34
tile_stride_height: 18
tile_stride_width: 16
```

支持将视频帧分块处理以降低显存占用（默认关闭）。

### 4. 视频与动作的分块结构

```yaml
num_frame_per_block: 1
num_action_per_block: 32
num_state_per_block: 1
frame_seqlen: 880
```

定义了视频帧、动作和状态在序列中的分块方式。每个 block 包含 1 帧视频、32 个动作 token、1 个状态 token。

### 5. 视觉-语言自注意力（VL Self-Attention）

```yaml
vl_self_attention_cfg:
  _target_: groot.vla.model.n1_5.modules.cross_attention_dit.SelfAttentionTransformer
  num_layers: 4
  num_attention_heads: 24
  attention_head_dim: 64
```

使用 4 层、24 头的自注意力 Transformer 来融合视觉和语言特征，作为动作预测的条件输入。

### 6. WAN 扩散模型（Diffusion Model）

```yaml
diffusion_model_cfg:
  _target_: groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk.CausalWanModel
  dim: 5120
  num_heads: 40
  num_layers: 40
  max_chunk_size: ${max_chunk_size}
```

核心扩散模型是一个修改版的 WAN DiT（Diffusion Transformer），具有 5120 维、40 层、40 头的大规模架构。使用因果分块注意力（Causal Chunk Attention）支持高效推理。

### 7. 预训练编码器

```yaml
text_encoder_cfg: ...    # WAN 文本编码器
image_encoder_cfg: ...   # WAN 图像编码器（用于 I2V）
vae_cfg: ...             # WAN 视频 VAE
```

文本编码器、图像编码器和 VAE 均来自 WAN 预训练模型，通过各自的 `_pretrained_path` 参数指定路径。

### 8. 噪声调度

```yaml
noise_beta_alpha: 1.5
noise_beta_beta: 1.0
noise_s: 0.999
num_timestep_buckets: 1000
```

使用 Beta 分布进行噪声时间步采样，通过 `alpha=1.5, beta=1.0` 的参数偏向较高噪声水平。

### 9. 解耦噪声采样

```yaml
decouple_video_action_noise: false
video_noise_beta_alpha: 3.0
video_noise_beta_beta: 1.0
```

支持将视频和动作的噪声采样解耦（默认关闭）。开启后，视频使用独立的 Beta 分布（偏向高噪声），动作使用均匀分布。

## 配置参数表格

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `add_pos_embed` | `true` | 是否添加位置编码 |
| `hidden_size` | `64` | 投影层隐藏维度 |
| `attn_dropout` | `0.2` | 注意力 Dropout |
| `repa_layer` | `8` | 表征对齐层数 |
| `repa_coeff` | `1.0` | 表征对齐损失系数 |
| `train_architecture` | `"lora"` | 训练架构（LoRA 微调） |
| `lora_rank` | `4` | LoRA 秩 |
| `lora_alpha` | `4` | LoRA alpha 缩放 |
| `lora_target_modules` | `"q,k,v,o,ffn.0,ffn.2"` | LoRA 目标模块 |
| `num_frames` | `${num_frames}` | 总帧数（来自数据配置） |
| `num_frame_per_block` | `1` | 每块帧数 |
| `num_action_per_block` | `32` | 每块动作 token 数 |
| `num_state_per_block` | `1` | 每块状态 token 数 |
| `frame_seqlen` | `880` | 帧序列长度 |
| `input_embedding_dim` | `1536` | 输入嵌入维度 |
| `use_gradient_checkpointing` | `true` | 启用梯度检查点（节省显存） |
| `use_vlln` | `true` | 使用视觉-语言-动作链接网络 |
| `num_inference_timesteps` | `4` | 推理时扩散步数 |
| `noise_beta_alpha` | `1.5` | 噪声 Beta 分布 alpha |
| `noise_beta_beta` | `1.0` | 噪声 Beta 分布 beta |
| `decouple_video_action_noise` | `false` | 是否解耦视频/动作噪声 |
| `tune_projector` | `true` | 是否训练投影层 |
| `tune_diffusion_model` | `true` | 是否训练扩散模型 |
| `action_loss_embodiment_ids` | `[26, 17, 32]` | 计算动作损失的机器人 ID |
| `max_state_dim` | `${max_state_dim}` | 最大状态维度 |
| `max_action_dim` | `${max_action_dim}` | 最大动作维度 |
| `action_dim` | `${max_action_dim}` | 动作维度 |
| `action_horizon` | `${action_horizon}` | 动作预测时间步 |

## 与其他配置/模块的关系

- **被引用**：由 `model/dreamzero/vla.yaml` 的 `defaults` 引入。
- **依赖的全局参数**：`num_frames`、`max_state_dim`、`max_action_dim`、`action_horizon`、`max_chunk_size` 等来自数据配置和 transform 配置。
- **预训练模型路径**：`dit_version`、`text_encoder_pretrained_path`、`image_encoder_pretrained_path`、`vae_pretrained_path` 需要用户在命令行或上层配置中指定。
- **backbone 关系**：`backbone_embedding_dim` 引用 `backbone_hidden_size`（来自 backbone 配置）。

## 总结

此配置文件定义了 DreamZero 最核心的模块 -- 基于 WAN 视频扩散模型的 Flow Matching 动作头。它将 WAN 的 40 层 DiT 架构与动作预测相融合，通过 LoRA 高效微调策略适配机器人控制任务。配置包含了扩散模型、文本/图像编码器、VAE、噪声调度、视觉-语言融合等完整的模块定义，是理解 DreamZero "视频生成驱动动作预测" 核心思想的关键文件。
