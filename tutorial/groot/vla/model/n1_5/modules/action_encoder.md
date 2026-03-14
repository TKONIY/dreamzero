# `action_encoder.py` -- 动作编码器（基于 Pi0）

## 文件概述

本文件实现了基于 Pi0 论文的动作编码器，将原始动作向量和扩散时间步融合编码为高维特征向量。该编码器是 flow matching 训练流程中的核心组件，负责将带噪声的动作和时间步信息编码为 DiT 模型可以处理的 token 序列。

## 关键代码解析

### swish 激活函数

```python
def swish(x):
    return x * torch.sigmoid(x)
```

Swish 激活函数（也称 SiLU），在扩散模型中广泛使用，提供了比 ReLU 更平滑的非线性变换。

### 正弦位置编码

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        timesteps = timesteps.float()
        B, T = timesteps.shape
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        enc = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        return enc
```

正弦位置编码将标量时间步转换为高维向量。编码公式基于经典的 Transformer 位置编码，使用不同频率的正弦和余弦函数，使模型能够区分不同的扩散阶段。

### ActionEncoder -- 动作编码器核心

```python
class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(action_dim, hidden_size)      # 动作投影
        self.W2 = nn.Linear(2 * hidden_size, hidden_size)  # 融合层
        self.W3 = nn.Linear(hidden_size, hidden_size)      # 输出层
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        B, T, _ = actions.shape
        timesteps = timesteps.unsqueeze(1).expand(-1, T)  # (B,) -> (B,T)
        a_emb = self.W1(actions)                           # 动作嵌入
        tau_emb = self.pos_encoding(timesteps)             # 时间步嵌入
        x = torch.cat([a_emb, tau_emb], dim=-1)           # 拼接
        x = swish(self.W2(x))                              # 融合 + 激活
        x = self.W3(x)                                     # 输出投影
        return x
```

编码流程如下：
1. **W1**: 将原始动作（维度 `action_dim`）投影到隐藏空间（维度 `hidden_size`）
2. **正弦编码**: 将标量扩散时间步编码为与隐藏空间同维度的向量
3. **拼接 + W2**: 将动作嵌入和时间步嵌入拼接（维度 `2*hidden_size`），通过 W2 压缩回 `hidden_size`
4. **W3**: 最终的线性投影层

这种设计确保了时间步信息和动作信息在特征空间中充分融合。

## 核心类/函数表格

| 类/函数名 | 功能描述 |
|-----------|---------|
| `swish()` | Swish/SiLU 激活函数 |
| `SinusoidalPositionalEncoding` | 将标量时间步编码为正弦位置向量 |
| `ActionEncoder` | 融合动作和时间步信息的编码器 |

## 与其他模块的关系

- **被 `wan_flow_matching_action_tf.py` 使用**: 作为动作头内部的编码组件
- **被 `wan_video_dit_action_casual_chunk.py` 引用**: `MultiEmbodimentActionEncoder` 扩展了此编码器以支持多机器人形态
- **输出供 DiT 模型消费**: 编码后的动作 token 与视频 token 一起送入 DiT 进行联合去噪

## 总结

`ActionEncoder` 是 DreamZero 中将机器人动作转换为神经网络可处理的 token 表示的核心模块。它借鉴了 Pi0 的设计，通过将动作和扩散时间步的编码进行拼接-融合，为后续的 flow matching 训练提供了信息丰富的动作表示。
