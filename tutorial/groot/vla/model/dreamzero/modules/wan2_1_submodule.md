# `wan2_1_submodule.py` -- Wan2.1 DiT 子模块集合

## 文件概述

本文件包含了 Wan2.1 DiT 模型的所有核心子模块，是 `wan_video_dit_action_casual_chunk.py` 使用的新版 DiT 实现的基础。与 `wan_video_dit.py` 中的早期实现不同，本文件使用了更高效的注意力实现和更规范的代码结构。

## 关键代码解析

### RoPE 相关函数

```python
def rope_params_polar(max_seq_len, dim, theta=10000):
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

def rope_action_apply_polar(x, freqs, freqs_action, freqs_state,
                             action_register_length, num_action_per_block, num_state_per_block):
    if action_register_length is not None:
        chunk_size = action_register_length // (num_action_per_block + num_state_per_block)
        freqs_1d_action = freqs_action[:chunk_size * num_action_per_block]
        freqs_1d_state = freqs_state[:chunk_size * num_state_per_block]
        freqs = torch.cat([freqs, freqs_1d_action, freqs_1d_state], dim=0)
    x = torch.view_as_real(x * freqs).flatten(3)
    return x
```

`rope_action_apply` 将 3D 视频 RoPE 和 1D 动作/状态 RoPE 拼接，使不同类型的 token 使用不同的位置编码策略。

### WanAttentionBlock -- 注意力块

```python
class WanAttentionBlock(nn.Module):
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, ...):
        self.self_attn = WanSelfAttention(dim, num_heads, ...)
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, ...)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        e = (self.modulation + e).chunk(6, dim=1)
        y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs)
        x = x + y * e[2]
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]
        return x
```

### I2V 交叉注意力

```python
class WanI2VCrossAttention(WanSelfAttention):
    def forward(self, x, context, crossattn_cache=None):
        context_img = context[:, :257]   # CLIP 图像特征
        context = context[:, 257:]        # 文本特征
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        # 文本交叉注意力
        k = self.norm_k(self.k(context)); v = self.v(context)
        x = flash_attention(q, k, v)
        # 图像交叉注意力（额外分支）
        k_img = self.norm_k_img(self.k_img(context_img))
        v_img = self.v_img(context_img)
        img_x = flash_attention(q, k_img, v_img)
        x = self.o(x + img_x)  # 融合文本和图像注意力
```

I2V 模式下，交叉注意力分为两支：文本注意力和图像注意力，各自独立计算后相加。

### WanModel -- 完整模型

```python
class WanModel(ModelMixin, ConfigMixin):
    def forward(self, x, t, context, seq_len, clip_fea=None, y=None, ...):
        # 1. Patch embedding + 位置编码
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        x = torch.cat([...])  # 填充到统一长度

        # 2. 时间步嵌入
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # 3. 条件编码
        context = self.text_embedding(torch.stack([...]))
        if clip_fea: context = torch.cat([self.img_emb(clip_fea), context], dim=1)

        # 4. Transformer 块
        for block in self.blocks:
            x = block(x, e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
                     freqs=self.freqs, context=context, context_lens=context_lens)

        # 5. 输出头 + unpatchify
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)
```

### RegisterTokens -- 分类 token

```python
class RegisterTokens(nn.Module):
    def __init__(self, num_registers, dim):
        self.register_tokens = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.rms_norm = WanRMSNorm(dim)
```

用于 GAN 判别器风格的分类分支。

## 核心类/函数表格

| 类/函数名 | 功能描述 |
|-----------|---------|
| `rope_params()` / `rope_apply()` | RoPE 参数计算和应用 |
| `rope_action_apply()` | 动作/状态 token 的 RoPE |
| `WanRMSNorm` | RMS 归一化 |
| `WanSelfAttention` | 自注意力 |
| `WanT2VCrossAttention` | 文本到视频交叉注意力 |
| `WanI2VCrossAttention` | 图像到视频交叉注意力 |
| `WanAttentionBlock` | 完整注意力块 |
| `WanModel` | 完整 Wan2.1 DiT 模型 |
| `Head` | 输出投影头 |
| `MLPProj` | CLIP 特征投影 |
| `RegisterTokens` | 分类 register token |

## 与其他模块的关系

- **被 `wan_video_dit_action_casual_chunk.py` 导入**: 使用 RoPE 函数和基础组件
- **使用 `attention.py`**: 底层注意力计算
- **与 `wan_video_dit.py` 功能类似**: 但代码更新、更高效

## 总结

`wan2_1_submodule.py` 包含了 Wan2.1 DiT 的全套子模块，是 DreamZero 中较新版本的 DiT 实现基础。它支持变长序列、I2V 交叉注意力、动作 token RoPE 等关键特性。
