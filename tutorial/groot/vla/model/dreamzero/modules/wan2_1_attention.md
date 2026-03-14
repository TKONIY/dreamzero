# `wan2_1_attention.py` -- Wan2.1 注意力模块

## 文件概述

本文件实现了 Wan2.1 的高级注意力模块 `AttentionModule`，支持多种注意力后端（Flash Attention 2/3、Transformer Engine、PyTorch SDPA、TensorRT），并提供了统一的接口。与 `attention.py` 中的函数式接口不同，这是一个面向对象的注意力模块，通过配置决定后端并在初始化时绑定。

## 关键代码解析

### flash_attention 函数（增强版）

```python
def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                    softmax_scale=None, q_scale=None, causal=False,
                    window_size=None, deterministic=False, dtype=torch.bfloat16,
                    version=None):
    # 支持变长序列（通过 q_lens/k_lens 参数）
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, ...)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # 构建 cumulative sequence lengths
    cu_seqlens_q = torch.cat([zeros, q_lens]).cumsum(0).to(torch.int32)
    cu_seqlens_k = torch.cat([zeros, k_lens]).cumsum(0).to(torch.int32)

    # 使用 varlen 版本的 flash attention
    x = flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)
```

相比 `attention.py`，增强版支持：Q 缩放（`q_scale`）、更严格的 GPU 检测（仅 H100 使用 FA3）。

### AttentionModule -- 统一注意力模块

```python
class AttentionModule(torch.nn.Module):
    def __init__(self, num_heads, head_dim, dropout_p=0., softmax_scale=None,
                 causal=False, window_size=None, dtype=torch.bfloat16, backend=None):
        # 自动选择后端
        if os.getenv("ATTENTION_BACKEND") is not None:
            backend = os.getenv("ATTENTION_BACKEND")
        else:
            backend = "FA2"

        if backend == "torch":
            self.attn_func = _torch_impl  # PyTorch SDPA
        elif backend == "TE":
            self.attn_backend = DotProductAttention(...)  # Transformer Engine
            self.attn_func = _te_impl
        elif backend in ("FA2", "FA3"):
            self.attn_func = _flash_attn_impl  # Flash Attention

    def forward(self, q, k, v, q_lens=None, k_lens=None):
        if self.backend in ("torch", "TE"):
            return self.attn_func(q, k, v)
        else:
            return self.attn_func(q, k, v, q_lens, k_lens)
```

关键设计：
- 通过环境变量 `ATTENTION_BACKEND` 选择后端
- 默认使用 FA2（Flash Attention 2）
- TensorRT 导出时回退到 PyTorch SDPA
- TE 不可用时自动回退到 FA2

## 核心类/函数表格

| 类/函数名 | 功能描述 |
|-----------|---------|
| `flash_attention()` | 增强版 flash attention 函数 |
| `AttentionModule` | 统一注意力模块（多后端） |

## 与其他模块的关系

- **被 `wan_video_dit_action_casual_chunk.py` 使用**: `CausalWanSelfAttention` 使用此模块
- **被 `wan2_1_submodule.py` 使用**: `WanSelfAttention` 等使用 `attention.py` 中的函数
- **使用 `cudnn_attention.py`**: 当 Transformer Engine 可用时使用 cuDNN 后端

## 总结

`wan2_1_attention.py` 提供了生产级的注意力计算模块，通过自动后端选择和统一接口，确保 DreamZero 在不同硬件环境下都能使用最优的注意力实现。
