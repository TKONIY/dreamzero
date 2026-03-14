# `attention.py` -- 基础注意力函数

## 文件概述

本文件提供了底层的注意力计算函数，支持 Flash Attention 2/3 和 PyTorch SDPA 后端。这是 `wan2_1_submodule.py` 中各注意力模块使用的基础计算层。

## 关键代码解析

### flash_attention -- 变长序列 Flash Attention

```python
def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                    softmax_scale=None, q_scale=None, causal=False,
                    window_size=(-1, -1), deterministic=False,
                    dtype=torch.bfloat16, version=None):
    # 预处理：将 batch 中变长序列打包
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # 计算累积序列长度（flash attention varlen 接口需要）
    cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)

    # 选择 FA3 或 FA2
    if FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)
    else:
        x = flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)
    return x.type(out_dtype)
```

### attention -- 带回退的通用注意力

```python
def attention(q, k, v, q_lens=None, k_lens=None, ...):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(q, k, v, ...)
    else:
        # 回退到 PyTorch SDPA
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, ...)
        return out
```

TensorRT 导出时使用 PyTorch SDPA，并进行相应的张量布局转换。

## 核心类/函数表格

| 函数名 | 功能描述 |
|--------|---------|
| `flash_attention()` | 变长序列 Flash Attention（FA2/FA3） |
| `attention()` | 通用注意力函数（自动选择后端） |

## 与其他模块的关系

- **被 `wan2_1_submodule.py` 使用**: `WanSelfAttention` 等调用 `flash_attention`
- **与 `wan2_1_attention.py` 功能互补**: 后者提供面向对象的 `AttentionModule`
- **与 `cudnn_attention.py` 配合**: 当使用 Transformer Engine 后端时

## 总结

`attention.py` 是 DreamZero 注意力计算的基础层，提供了高效的变长序列 Flash Attention 实现和多后端回退机制。
