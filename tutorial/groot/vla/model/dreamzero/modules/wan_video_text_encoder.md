# `wan_video_text_encoder.py` -- Wan 文本编码器（UMT5）

## 文件概述

本文件实现了 Wan2.1 使用的文本编码器，基于 T5 架构（具体为 UMT5-XXL）。文本编码器将自然语言指令编码为条件向量，指导视频生成模型生成符合指令描述的视频和动作。

## 关键代码解析

### T5 核心组件

```python
class T5Attention(nn.Module):
    def __init__(self, dim, dim_attn, num_heads, dropout=0.1):
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)

    def forward(self, x, context=None, mask=None, pos_bias=None):
        # T5 不使用 scaling（与标准 Transformer 不同）
        attn = torch.einsum('binc,bjnc->bnij', q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum('bnij,bjnc->binc', attn, v)
        return self.o(x)
```

T5 注意力的独特之处：不使用 `1/sqrt(d)` 缩放，而是直接计算 QK 点积。

### T5RelativeEmbedding -- 相对位置编码

```python
class T5RelativeEmbedding(nn.Module):
    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, rel_pos):
        # 对数分桶：近距离精细区分，远距离粗略区分
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (torch.log(rel_pos / max_exact) /
                                     math.log(max_dist / max_exact) * (num_buckets - max_exact))
```

T5 使用对数分桶的相对位置编码，近距离的位置被精细区分，远距离的位置被粗略归并。

### WanTextEncoder -- 文本编码器

```python
class WanTextEncoder(torch.nn.Module):
    def __init__(self, vocab=256384, dim=4096, dim_attn=4096, dim_ffn=10240,
                 num_heads=64, num_layers=24, num_buckets=32, ...):
        self.token_embedding = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets, ...)
            for _ in range(num_layers)
        ])
        self.norm = T5LayerNorm(dim)

    def forward(self, ids, mask=None):
        x = self.token_embedding(ids)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        x = self.norm(x)
        return x  # [B, L, 4096]
```

默认配置为 UMT5-XXL 规模：4096 维隐藏层、64 个注意力头、24 层 Transformer，输出 4096 维的文本特征。

## 核心类/函数表格

| 类/函数名 | 功能描述 |
|-----------|---------|
| `T5LayerNorm` | T5 风格的 RMS 归一化 |
| `T5Attention` | T5 注意力（无缩放） |
| `T5FeedForward` | T5 门控前馈网络 |
| `T5SelfAttention` | T5 自注意力块 |
| `T5RelativeEmbedding` | 对数分桶相对位置编码 |
| `WanTextEncoder` | 完整的 UMT5 文本编码器 |

## 与其他模块的关系

- **被 `wan_flow_matching_action_tf.py` 使用**: 编码文本指令作为条件
- **输出供 `wan_video_dit.py` / `wan2_1_submodule.py` 消费**: 通过 `text_embedding` 投影后作为交叉注意力的条件
- **由 `dreamzero_cotrain.py` 的分词器预处理**: 文本先经分词器编码为 token ID

## 总结

`wan_video_text_encoder.py` 实现了基于 UMT5-XXL 的文本编码器，将自然语言机器人指令编码为 4096 维的条件特征序列。这些特征通过 DiT 的交叉注意力机制指导视频和动作的联合生成，使模型能够根据语言指令生成对应的操作视频和控制信号。
