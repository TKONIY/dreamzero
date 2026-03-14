# `wan_video_dit_action_casual_chunk.py` -- 因果式动作-视频联合 DiT

## 文件概述

本文件是 DreamZero 最核心的 DiT 变体实现，扩展了标准 Wan DiT 以支持：

1. **动作 token 嵌入**: 将机器人动作编码为 token 与视频 token 联合处理
2. **因果注意力**: 支持自回归式的逐块生成（每块包含若干帧+对应动作+状态）
3. **Teacher Forcing 训练**: 将序列分为 clean half 和 noisy half 以稳定训练
4. **多机器人形态**: 通过 `CategorySpecificMLP` 支持不同机器人的动作编码/解码
5. **块级因果注意力掩码**: 每个时间块只能看到之前的块（因果性）

## 关键代码解析

### CategorySpecificLinear / CategorySpecificMLP -- 多形态线性层

```python
class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]  # 按形态 ID 选择权重
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)
```

为每种机器人形态维护独立的线性变换权重，通过 `cat_ids` 索引选择对应的权重矩阵，实现多形态的动作编码。

### MultiEmbodimentActionEncoder -- 多形态动作编码器

```python
class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        a_emb = self.W1(actions, cat_ids)
        tau_emb = self.pos_encoding(timesteps)
        x = swish(self.W2(torch.cat([a_emb, tau_emb], dim=-1), cat_ids))
        x = self.W3(x, cat_ids)
        return x
```

扩展了基础 `ActionEncoder`，每个线性层都替换为 `CategorySpecificLinear`，使得不同机器人的动作空间可以有独立的编码映射。

### CausalWanSelfAttention -- 因果自注意力

这是整个文件最复杂的类，实现了块级因果注意力掩码。注意力模式为：

```
序列结构: [first_image][image_blocks][action_blocks][state_blocks]

注意力规则:
- first_image: 仅自注意力
- image_block[i]: 可看到 first_image + image_blocks[0:i] + action_block[i] + state_block[i]
- action_block[i]: 可看到 first_image + image_blocks[0:i] + image_block[i] + action_block[i] + state_block[i]
- state_block[i]: 仅自注意力
```

在 Teacher Forcing 模式下，序列被分为 clean half（无噪声）和 noisy half（带噪声），各自维持上述注意力模式：

```python
def forward(self, x, freqs, freqs_action, freqs_state, action_register_length, ...):
    if is_tf:  # Teacher Forcing
        q_context = q[:, :(s-action_register_length)//2]  # clean half
        q_noisy = q[:, (s-action_register_length)//2:]     # noisy half

        # Clean half: 标准因果注意力
        clean_image_outputs = self._process_clean_image_only(...)

        # Noisy half: 可以看到 clean half 作为条件
        noisy_image_outputs = self._process_noisy_image_blocks(...)
        noisy_action_outputs = self._process_noisy_action_blocks(...)
        noisy_state_outputs = self._process_state_blocks(...)

        x = torch.cat([clean_outputs, noisy_outputs], dim=1)
```

### causal_rope_action_apply -- 因果 RoPE 应用

```python
def causal_rope_action_apply_polar(x, freqs, freqs_action, freqs_state,
                                     action_register_length, num_action_per_block,
                                     num_state_per_block, action_state_index):
    # 为当前帧索引选择对应的动作/状态频率
    freqs_action = freqs_action[action_state_index * num_action_per_block:
                                 (action_state_index + 1) * num_action_per_block]
    freqs_state = freqs_state[action_state_index * num_state_per_block:
                               (action_state_index + 1) * num_state_per_block]
    freqs_1d = torch.cat([freqs_action, freqs_state], dim=0)
    freqs = torch.cat([freqs, freqs_1d], dim=0)
    # 应用旋转位置编码
    x = torch.view_as_real(x * freqs).flatten(3)
    return x
```

为动作和状态 token 使用独立的 1D RoPE（区别于视频 token 的 3D RoPE），使模型能够区分不同帧对应的动作/状态。

## 核心类/函数表格

| 类/函数名 | 功能描述 |
|-----------|---------|
| `CategorySpecificLinear` | 按形态 ID 选择权重的线性层 |
| `CategorySpecificMLP` | 多形态 MLP |
| `MultiEmbodimentActionEncoder` | 多形态动作编码器 |
| `CausalWanSelfAttention` | 块级因果自注意力 |
| `_blockwise_causal_flash_attn()` | 块级因果 flash attention |
| `_process_clean_image_only()` | 处理 clean half 的图像 token |
| `_process_noisy_image_blocks()` | 处理 noisy half 的图像 token |
| `_process_noisy_action_blocks()` | 处理 noisy half 的动作 token |
| `_process_state_blocks()` | 处理状态 token |
| `causal_rope_action_apply()` | 动作/状态 token 的 RoPE |

## 与其他模块的关系

- **扩展 `wan_video_dit.py`**: 在其基础上增加动作 token 和因果注意力
- **使用 `wan2_1_attention.py`**: 使用 `AttentionModule` 作为注意力后端
- **使用 `wan2_1_submodule.py`**: 使用 `WanRMSNorm`、`rope_action_apply` 等基础组件
- **使用 `action_encoder.py`**: 使用 `SinusoidalPositionalEncoding` 和 `swish`
- **被 `wan_flow_matching_action_tf.py` 使用**: 作为联合视频-动作生成的 DiT 主干

## 总结

本文件是 DreamZero "世界模型 + 策略模型"统一架构的核心实现。通过在 DiT 的自注意力中引入块级因果掩码和动作/状态 token，模型可以在单次前向传播中同时处理视频和动作的去噪，实现了视频生成与动作预测的深度融合。Teacher Forcing 的 clean/noisy 分割设计大大稳定了训练过程。
