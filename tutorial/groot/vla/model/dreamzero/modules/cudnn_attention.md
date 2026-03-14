# `cudnn_attention.py` -- cuDNN 融合注意力

## 文件概述

本文件实现了基于 NVIDIA Transformer Engine 和 cuDNN 的融合注意力（Fused Attention），是性能最优的注意力后端。它通过 `torch.library.custom_op` 注册自定义算子，支持 `torch.compile` 和自动微分。

## 关键代码解析

### DotProductAttention -- 点积注意力封装

```python
class DotProductAttention(torch.nn.Module):
    def __init__(self, num_attention_heads, kv_channels, qkv_format="bshd",
                 attn_mask_type="no_mask", window_size=None, attention_dropout=0.0):
        self.fused_attention = FusedAttention(
            self.softmax_scale, deterministic=False, attention_dropout=self.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer):
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * max_seqlen_q, step=max_seqlen_q, ...)
        return self.fused_attention(query_layer, key_layer, value_layer,
                                     cu_seqlens_q=cu_seqlens_q, ...)
```

### fused_attn -- 自定义算子

```python
@torch.library.custom_op("groot::fused_attn", mutates_args=())
def fused_attn(is_training, max_seqlen_q, max_seqlen_kv, cu_seqlens_q, cu_seqlens_kv,
               q, k, v, fake_dtype, window_size, ...):
    output_tensors = tex.fused_attn_fwd(*args)  # 调用 TE C++ 后端
    return output_tensors
```

通过 `torch.library.custom_op` 注册自定义前向算子，并通过 `register_fake` 和 `register_autograd` 分别注册 FakeTensor 推导和反向传播逻辑，使其兼容 `torch.compile(fullgraph=True)`。

### 反向传播

```python
@torch.library.custom_op("groot::fused_attn_bwd_op", mutates_args=())
def fused_attn_bwd_op(max_seqlen_q, max_seqlen_kv, ...):
    dq, dk, dv, *rest = tex.fused_attn_bwd(*args)
    return dq, dk, dv

fused_attn.register_autograd(fused_attn_bwd_impl, setup_context=fused_attn_bwd_setup_context)
```

## 核心类/函数表格

| 类/函数名 | 功能描述 |
|-----------|---------|
| `DotProductAttention` | 点积注意力封装 |
| `FusedAttention` | cuDNN 融合注意力 |
| `fused_attn` | 自定义前向算子 |
| `fused_attn_bwd_op` | 自定义反向算子 |
| `prepare_for_saving()` / `restore_from_saved()` | 张量保存/恢复辅助函数 |

## 与其他模块的关系

- **被 `wan2_1_attention.py` 使用**: 当 `backend="TE"` 时使用此模块
- **依赖 `transformer_engine`**: 需要安装 NVIDIA Transformer Engine

## 总结

`cudnn_attention.py` 提供了最高性能的注意力计算后端，通过 cuDNN 的融合内核将 QKV 投影、注意力计算和输出投影融合为单个 GPU 内核调用，显著减少内存访问和内核启动开销。
