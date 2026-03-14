# `wan_video_vae.py` -- Wan 视频 VAE（变分自编码器）

## 文件概述

本文件实现了 Wan2.1 的 3D 视频 VAE，负责在像素空间和潜在空间之间转换视频。VAE 将高分辨率视频压缩为低维潜在表示以便 DiT 处理，推理后再将潜在表示解码回像素空间。文件包含两个版本的编解码器（标准版和 3.8 版）以及支持分块处理的缓存机制。

## 关键代码解析

### CausalConv3d -- 因果 3D 卷积

```python
class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        if cache_x is not None and self._padding[4] > 0:
            x = torch.cat([cache_x, x], dim=2)
        x = F.pad(x, padding)
        return super().forward(x)
```

因果 3D 卷积确保时间维度上只使用过去和当前帧的信息（不看未来帧），支持缓存上一块的帧以实现流式处理。

### Encoder3d -- 3D 编码器

```python
class Encoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], ...):
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)
        # 下采样块: 残差块 + 注意力 + 空间/时间下采样
        # 中间块: 残差 + 注意力 + 残差
        # 输出: RMS归一化 + SiLU + 卷积

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x = self.conv1(x)
        for layer in self.downsamples:
            x = layer(x, feat_cache, feat_idx)
        for layer in self.middle:
            x = layer(x)
        for layer in self.head:
            x = layer(x)
        return x
```

编码器通过多级下采样将视频从像素空间压缩到潜在空间。`temperal_downsample` 控制哪些层在时间维度上下采样。

### Decoder3d -- 3D 解码器

解码器结构与编码器对称，通过上采样将潜在表示还原为视频。

### VideoVAE_ -- 完整 VAE

```python
class VideoVAE_(nn.Module):
    def encode(self, x, scale):
        # 分块编码（每块4帧，首帧单独处理）
        out = self.encoder(x[:, :, :1, :, :], feat_cache=feat_map)
        for i in range(1, iter_):
            out_ = self.encoder(x[:, :, 1+4*(i-1):1+4*i, :, :], feat_cache=feat_map)
            out = torch.cat([out, out_], dim=2)
        mu, _ = self.conv1(out).chunk(2, dim=1)
        mu = (mu - scale[0]) * scale[1]  # 标准化
        return mu

    def decode(self, z, scale):
        # 反标准化 + 分块解码
        z = z / scale[1] + scale[0]
        out = self.decoder(z[:, :, :1, :, :], feat_cache=feat_map)
        for i in range(1, iter_):
            out_ = self.decoder(z[:, :, 1+2*(i-1):1+2*i, :, :], feat_cache=feat_map)
            out = torch.cat([out, out_], dim=2)
        return out.clamp(-1, 1)
```

VAE 的编解码支持分块处理：编码时每 4 帧一块，解码时每 2 帧一块，通过 `feat_cache` 在块之间传递因果卷积的缓存状态，实现内存友好的长视频处理。

### 3.8 版本

文件还包含 `Encoder3d_38`、`Decoder3d_38` 等 3.8 版本的变体，使用了更先进的 `AvgDown3D`（平均下采样）和 `DupUp3D`（复制上采样）替代传统的卷积上下采样。

## 核心类/函数表格

| 类/函数名 | 功能描述 |
|-----------|---------|
| `CausalConv3d` | 因果 3D 卷积，支持缓存 |
| `ResidualBlock` | 残差块，含因果卷积 |
| `AttentionBlock` | 2D 自注意力块 |
| `Resample` | 2D/3D 上下采样 |
| `Encoder3d` | 3D 视频编码器 |
| `Decoder3d` | 3D 视频解码器 |
| `VideoVAE_` | 完整 VAE（编码 + 解码） |
| `Encoder3d_38` / `Decoder3d_38` | 3.8 版编解码器 |
| `AvgDown3D` / `DupUp3D` | 平均下采样 / 复制上采样 |

## 与其他模块的关系

- **被 `wan_flow_matching_action_tf.py` 使用**: `encode_video()` 和 `decode_video()` 调用 VAE
- **与 `wan_video_dit.py` 配合**: DiT 在 VAE 的潜在空间中工作
- **压缩比**: 空间 8x 下采样，时间 4x 下采样（首帧单独处理）

## 总结

`wan_video_vae.py` 实现了 Wan2.1 的 3D 视频 VAE，将高分辨率视频压缩为低维潜在表示。通过因果卷积和分块处理机制，VAE 支持长视频的高效编解码。在 DreamZero 中，VAE 负责将机器人操作视频转换为 DiT 可处理的潜在空间表示，是像素级视觉信息与模型内部表示之间的桥梁。
