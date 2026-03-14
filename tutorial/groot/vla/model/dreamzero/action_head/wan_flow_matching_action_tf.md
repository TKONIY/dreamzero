# `wan_flow_matching_action_tf.py` -- 基于 Wan 视频模型的 Flow Matching 动作头

## 文件概述

本文件是 DreamZero 系统中最核心、最复杂的模块之一，实现了基于 Wan2.1 视频生成模型的 Flow Matching 动作预测头。它将视频生成（扩散模型）与机器人动作预测统一在同一个 DiT（Diffusion Transformer）架构中，实现了"世界模型 + 策略模型"的联合学习。

核心创新点：
1. 将机器人动作嵌入到视频扩散模型的 token 序列中
2. 使用 Flow Matching 训练范式同时去噪视频和动作
3. 支持 Teacher Forcing 训练和因果式自回归推理
4. 集成 LoRA 微调、VRAM 管理和分布式训练

## 关键代码解析

### WanFlowMatchingActionTFConfig -- 配置类

配置类包含了大量参数，关键的有：

- `diffusion_model_pretrained_path`: Wan2.1 DiT 预训练权重路径
- `text_encoder_pretrained_path`: UMT5 文本编码器路径
- `image_encoder_pretrained_path`: CLIP 图像编码器路径
- `vae_pretrained_path`: VAE 编码/解码器路径
- `action_horizon`, `action_dim`: 动作预测范围和维度
- `num_action_per_block`, `num_state_per_block`: 每个时间块的动作/状态 token 数
- `lora_rank`, `lora_target_modules`: LoRA 微调配置

### WanFlowMatchingActionTF -- 核心动作头

```python
class WanFlowMatchingActionTF(ActionHead):
    def __init__(self, config: WanFlowMatchingActionTFConfig):
        # 初始化四大组件
        self.model = WanModel(...)          # DiT 主干网络
        self.text_encoder = WanTextEncoder(...)  # 文本编码器
        self.image_encoder = WanImageEncoder(...)  # 图像编码器
        self.vae = WanVideoVAE(...)         # 视频 VAE

        # 动作编码/解码
        self.action_encoder = MultiEmbodimentActionEncoder(...)
        self.action_projector_in = CategorySpecificMLP(...)
        self.action_projector_out = CategorySpecificMLP(...)
        self.state_projector_in = CategorySpecificMLP(...)
```

### 训练流程 (forward)

```python
def forward(self, backbone_output, action_input):
    # 1. 编码视频为潜在空间
    latent_video = self.encode_video(video)  # [B, C, T, H, W]

    # 2. 编码文本和图像条件
    context = self.text_encoder(text_ids)
    clip_feature = self.image_encoder.encode_image(first_frame)

    # 3. 采样噪声时间步
    timestep = torch.rand(B) * num_train_timesteps

    # 4. 对视频潜在表示添加噪声
    noisy_latent = scheduler.add_noise(latent_video, noise, timestep)

    # 5. 对动作添加噪声（Flow Matching）
    noisy_action = (1 - sigma) * action + sigma * action_noise

    # 6. 编码带噪声的动作为 token
    action_tokens = self.action_encoder(noisy_action, timestep, embodiment_id)

    # 7. DiT 联合去噪：视频 + 动作
    model_output = self.model(noisy_latent, timestep, context, ...)

    # 8. 计算损失
    video_loss = MSE(model_output_video, target_video)
    action_loss = MSE(model_output_action, target_action)
    loss = action_loss_weight * action_loss + video_loss_weight * video_loss
```

### 推理流程 (lazy_joint_video_action)

```python
def lazy_joint_video_action(self, backbone_output, action_input, latent_video=None):
    # 1. 初始化纯噪声
    noise = torch.randn(latent_shape)
    action_noise = torch.randn(action_shape)

    # 2. 多步去噪循环
    for timestep in scheduler.timesteps:
        # 编码带噪声的动作
        action_tokens = self.action_encoder(noisy_action, timestep, ...)

        # DiT 预测噪声
        model_output = self.model(noisy_latent, timestep, context, ...)

        # 分离视频和动作输出
        video_output, action_output = split(model_output)

        # 一步去噪
        noisy_latent = scheduler.step(video_output, timestep, noisy_latent)
        noisy_action = (1 - sigma_next) * action_pred + sigma_next * action_noise

    # 3. VAE 解码视频
    video = self.vae.decode(denoised_latent)

    return {"action_pred": denoised_action, "video_pred": video}
```

### VRAM 管理

```python
def enable_vram_management(self):
    enable_vram_management(
        self.model,
        module_map={nn.Linear: AutoWrappedLinear, nn.LayerNorm: WanAutoCastLayerNorm},
        module_config={
            "offload_dtype": torch.bfloat16,
            "offload_device": "cpu",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
    )
```

通过 VRAM 管理机制，模型权重可以存储在 CPU 上，仅在计算时加载到 GPU，从而在有限显存下运行 14B 参数的 DiT 模型。

## 核心类/函数表格

| 类/函数名 | 功能描述 |
|-----------|---------|
| `WanFlowMatchingActionTFConfig` | 动作头完整配置 |
| `WanFlowMatchingActionTF` | 核心动作头类 |
| `forward()` | 训练时的前向传播（计算损失） |
| `get_action()` | 仅预测动作（不生成视频） |
| `joint_video_action()` | 联合生成视频和动作 |
| `lazy_joint_video_action()` | 惰性联合生成（节省显存） |
| `lazy_joint_video_action_causal_gt_cond()` | 因果式生成（使用真实条件帧） |
| `encode_video()` | 通过 VAE 编码视频到潜在空间 |
| `enable_vram_management()` | 启用 VRAM 管理以节省显存 |
| `inject_lora_after_loading()` | 在加载预训练权重后注入 LoRA |
| `hf_download()` / `ensure_file()` | 从 HuggingFace 下载模型文件 |

## 与其他模块的关系

- **继承 `base_action_head.py`**: 实现 `ActionHead` 接口
- **使用 `wan_video_dit.py` / `wan_video_dit_action_casual_chunk.py`**: DiT 主干网络
- **使用 `wan_video_text_encoder.py`**: UMT5 文本编码器
- **使用 `wan_video_image_encoder.py`**: CLIP 图像编码器
- **使用 `wan_video_vae.py`**: 视频 VAE 编解码器
- **使用 `flow_match_scheduler.py`**: Flow Matching 调度器
- **使用 `vram_management.py`**: 显存管理机制
- **被 `base_vla.py` 的 VLA 类调用**: 作为 action_head 组件

## 总结

`wan_flow_matching_action_tf.py` 是 DreamZero 的核心引擎，它将 Wan2.1 视频生成模型改造为联合的世界模型+策略模型。通过将动作 token 嵌入到视频扩散过程中，模型可以同时学习物理世界的动态规律和最优控制策略。这种设计使得模型既能生成未来视频（想象力），又能直接输出机器人动作（执行力），实现了"先想后做"的决策范式。
