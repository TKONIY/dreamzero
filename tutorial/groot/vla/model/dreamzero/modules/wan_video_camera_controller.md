# `wan_video_camera_controller.py` -- 相机控制模块

## 文件概述

本文件实现了 Wan2.1 的相机控制功能，通过 Plucker 坐标系（射线方向 + 射线力矩）编码相机运动，将相机轨迹信息注入 DiT 模型以控制生成视频的相机视角。

## 关键代码解析

### SimpleAdapter -- 相机条件注入适配器

```python
class SimpleAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, num_residual_blocks=1):
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=8)
        self.conv = nn.Conv2d(in_dim * 64, out_dim, kernel_size=kernel_size, stride=stride)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_dim) for _ in range(num_residual_blocks)]
        )

    def forward(self, x):
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).view(bs * f, c, h, w)
        x = self.pixel_unshuffle(x)   # 空间降采样 8x
        x = self.conv(x)               # 通道映射
        x = self.residual_blocks(x)    # 特征精炼
        return out.permute(0, 2, 1, 3, 4)  # 恢复帧维度
```

SimpleAdapter 将 Plucker 坐标编码的相机信息（6通道/帧）转换为与 DiT patch embedding 兼容的特征图，通过加法注入到 DiT 的输入中。

### Plucker 坐标系

```python
def ray_condition(K, c2w, H, W, device):
    # 从相机内参和外参计算射线方向和原点
    directions = torch.stack((xs, ys, zs), dim=-1)
    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # 射线方向
    rays_o = c2w[..., :3, 3]                                    # 射线原点
    rays_dxo = torch.linalg.cross(rays_o, rays_d)               # 射线力矩
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)             # 6D Plucker 坐标
    return plucker  # [B, V, H, W, 6]
```

Plucker 坐标用 6 个值（3D 射线方向 + 3D 射线力矩）表示每个像素的相机射线，是一种紧凑且几何等变的相机表示。

### generate_camera_coordinates -- 相机轨迹生成

```python
def generate_camera_coordinates(direction, length, speed=1/54, origin=...):
    coordinates = [list(origin)]
    while len(coordinates) < length:
        coor = coordinates[-1].copy()
        if "Left" in direction:  coor[9] += speed
        if "Right" in direction: coor[9] -= speed
        if "Up" in direction:    coor[13] += speed
        if "Down" in direction:  coor[13] -= speed
        coordinates.append(coor)
    return coordinates
```

支持 8 个方向的匀速相机运动生成。

## 核心类/函数表格

| 类/函数名 | 功能描述 |
|-----------|---------|
| `SimpleAdapter` | 相机条件注入适配器 |
| `ResidualBlock` | 2D 残差块 |
| `Camera` | 相机参数封装 |
| `ray_condition()` | 计算 Plucker 射线坐标 |
| `process_pose_file()` | 处理相机位姿文件 |
| `generate_camera_coordinates()` | 生成匀速相机轨迹 |

## 与其他模块的关系

- **被 `wan_video_dit.py` 使用**: `control_adapter` 将相机条件注入 DiT
- **被 `wan_flow_matching_action_tf.py` 间接使用**: 通过 DiT 模型的配置启用

## 总结

相机控制模块通过 Plucker 坐标编码相机运动，使 DreamZero 能够生成具有特定视角变化的视频。虽然在机器人操作场景中相机通常固定，但该模块为第一人称视角的机器人操作和多视角融合提供了基础能力。
