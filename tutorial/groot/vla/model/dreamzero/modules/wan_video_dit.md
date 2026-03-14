# `wan_video_dit.py` -- Wan 视频扩散 Transformer（DiT）

## 文件概述

本文件实现了 Wan2.1 的 Diffusion Transformer (DiT) 模型，是 DreamZero 视频生成和动作预测的核心主干网络。DiT 在潜在空间中对视频进行去噪，使用 3D 旋转位置编码 (RoPE)、自注意力、交叉注意力和自适应调制 (AdaLN) 等技术。

该文件包含了早期版本的 DiT 实现（使用 `einops` 和原版 flash attention），主要用于与旧版权重兼容和状态字典转换。

## 关键代码解析

### flash_attention -- 多后端注意力

```python
def flash_attention(q, k, v, num_heads, compatibility_mode=False):
    if compatibility_mode:
        x = F.scaled_dot_product_attention(q, k, v)
    elif FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_func(q, k, v)
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_func(q, k, v)
    elif SAGE_ATTN_AVAILABLE:
        x = sageattn(q, k, v)
    else:
        x = F.scaled_dot_product_attention(q, k, v)
    return x
```

自动选择最优的注意力后端，按优先级：Flash Attention 3 > Flash Attention 2 > SageAttention > PyTorch SDPA。

### 3D 旋转位置编码 (RoPE)

```python
class RotaryPositionEmbeddingWithPolarOp(nn.Module):
    def precompute_freqs_cis_3d(self, dim, end=1024, theta=10000.0):
        f_freqs_cis = self.precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
        h_freqs_cis = self.precompute_freqs_cis(dim // 3, end, theta)
        w_freqs_cis = self.precompute_freqs_cis(dim // 3, end, theta)
        return {"f": f_freqs_cis, "h": h_freqs_cis, "w": w_freqs_cis}

    def forward(self, f, h, w, a):
        freqs = torch.cat([
            self.freqs["f"][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs["h"][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs["w"][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1)
        return freqs
```

3D RoPE 将帧（f）、高度（h）、宽度（w）三个维度的频率分别编码，然后拼接为统一的位置编码。维度分配为：帧维度占 `dim - 2*(dim//3)`，高度和宽度各占 `dim//3`。

### DiTBlock -- 扩散 Transformer 块

```python
class DiTBlock(nn.Module):
    def forward(self, x, context, t_mod, freqs):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation + t_mod).chunk(6, dim=1)
        # 自注意力（带 AdaLN 调制）
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        # 交叉注意力
        x = x + self.cross_attn(self.norm3(x), context)
        # FFN（带 AdaLN 调制）
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x
```

每个 DiTBlock 包含：
1. **AdaLN 调制的自注意力**：时间步信息通过 shift/scale/gate 六个调制参数注入
2. **交叉注意力**：与文本/图像条件特征交互
3. **AdaLN 调制的 FFN**：同样受时间步调制的前馈网络

### WanModel -- 完整 DiT 模型

```python
class WanModel(ModelMixin, ConfigMixin):
    def forward(self, x, timestep, context, clip_feature=None, y=None, ...):
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # 拼接条件图像
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        x, (f, h, w) = self.patchify(x)
        freqs = self.rope(f=f, h=h, w=w, a=x.shape[1])
        for block in self.blocks:
            x = block(x, context, t_mod, freqs)
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
```

### WanModelStateDictConverter -- 权重转换器

文件还包含了从 diffusers 格式和 civitai 格式转换权重的工具，通过 `hash_state_dict_keys` 自动识别模型配置。

## 核心类/函数表格

| 类/函数名 | 功能描述 |
|-----------|---------|
| `flash_attention()` | 多后端自动选择的注意力计算 |
| `modulate()` | AdaLN 调制操作 |
| `sinusoidal_embedding_1d()` | 1D 正弦时间步编码 |
| `RotaryPositionEmbedding` | 3D 旋转位置编码 |
| `SelfAttention` | 带 RoPE 的自注意力 |
| `CrossAttention` | 交叉注意力（支持图像输入） |
| `DiTBlock` | 完整的 DiT 块 |
| `WanModel` | 完整的 Wan DiT 模型 |
| `WanModelStateDictConverter` | 权重格式转换器 |

## 与其他模块的关系

- **被 `wan_flow_matching_action_tf.py` 使用**: 作为 DiT 主干网络
- **与 `wan_video_dit_action_casual_chunk.py` 互补**: 后者是支持动作 token 和因果注意力的扩展版本
- **使用 `wan_video_camera_controller.py`**: `SimpleAdapter` 用于相机控制条件注入
- **使用 `utils.py`**: `hash_state_dict_keys` 用于模型配置自动识别

## 总结

`wan_video_dit.py` 是 Wan2.1 视频扩散模型的核心实现，提供了完整的 DiT 架构，包括 3D 补丁化、旋转位置编码、自适应调制注意力等关键组件。它是 DreamZero 视频生成能力的基础。
