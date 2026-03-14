# 视频变换 (`groot/vla/data/transform/video.py`)

## 文件概述

`video.py` 实现了丰富的视频帧预处理和数据增强变换。它定义了抽象基类 `VideoTransform` 以及多种具体变换（裁剪、缩放、旋转、翻转、颜色抖动、灰度化等），同时支持 `torchvision` 和 `albumentations` 两种后端。该文件约 1000 行代码，是变换模块中最大的文件之一。

## 关键代码解析

### 1. `VideoTransform` 抽象基类

```python
class VideoTransform(ModalityTransform):
    backend: str = "torchvision"  # 后端选择
    _train_transform: Callable | None     # 训练时变换
    _eval_transform: Callable | None      # 评估时变换
    _original_resolutions: dict[str, tuple[int, int]]  # 原始分辨率
```

#### 插值模式映射

通过 `_INTERPOLATION_MAP` 字典统一了不同后端的插值模式名称：

```python
_INTERPOLATION_MAP = {
    "nearest": {"albumentations": cv2.INTER_NEAREST, "torchvision": T.InterpolationMode.NEAREST},
    "linear": {"albumentations": cv2.INTER_LINEAR, "torchvision": T.InterpolationMode.BILINEAR},
    "cubic": {"albumentations": cv2.INTER_CUBIC, "torchvision": T.InterpolationMode.BICUBIC},
    # ...
}
```

#### `set_metadata` 方法

从数据集元数据中提取视频分辨率信息，并初始化训练/评估变换：

```python
def set_metadata(self, dataset_metadata):
    for key in self.apply_to:
        sub_key = key.split(".")[1]
        self.original_resolutions[key] = dataset_metadata.modalities.video[sub_key].resolution
    train_transform = self.get_transform(mode="train")
    eval_transform = self.get_transform(mode="eval")
```

#### `apply` 方法

处理多视图和批量数据：

1. 将所有视图沿第一维拼接
2. 如果是批量数据，使用 `einops.rearrange` 重塑形状
3. 根据后端应用变换（torchvision 直接应用，albumentations 使用 `ReplayCompose` 保证帧间一致性）
4. 将结果拆分回各视图

```python
# albumentations 帧间一致性保证
transformed = transform(image=first_frame)
replay_data = transformed["replay"]
transformed_frames = [
    transform.replay(replay_data, image=frame)["image"] for frame in views[1:]
]
```

### 2. `VideoCrop`

```python
class VideoCrop(VideoTransform):
    scale: float   # 裁剪比例
```

训练时使用 `RandomCrop`，评估时使用 `CenterCrop`。裁剪尺寸由原始分辨率和 `scale` 参数决定：

```python
size = (int(height * scale), int(width * scale))
```

### 3. `VideoResize`

```python
class VideoResize(VideoTransform):
    height: int      # 目标高度
    width: int       # 目标宽度
    interpolation: str = "linear"
    antialias: bool = True
```

训练和评估使用相同的缩放参数。

### 4. `VideoToTensor`

将 numpy 数组（uint8, [T, H, W, C]）转换为 PyTorch 张量（float32, [T, C, H, W]）：

```python
@staticmethod
def to_tensor(frames, output_on_cuda):
    frames = torch.from_numpy(frames)
    frames = frames.to(torch.float32) / 255.0
    return frames.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
```

### 5. `VideoToNumpy`

`VideoToTensor` 的逆操作：

```python
@staticmethod
def to_numpy(frames):
    frames = (frames.permute(0, 2, 3, 1) * 255).to(torch.uint8)
    return frames.cpu().numpy()
```

### 6. `VideoRandomErasing`

随机遮挡视频中的矩形区域，防止过拟合：

```python
class VideoRandomErasing(VideoTransform):
    probability: float = 0.2
    scale: tuple[float, float] = (0.02, 0.33)
    value: str | tuple = "random"  # 遮挡填充值
```

仅在训练时应用，评估时返回 `None`。

### 7. 颜色增强变换

| 类名 | 功能 | 训练/评估 |
|------|------|-----------|
| `VideoColorJitter` | 亮度/对比度/饱和度/色相抖动 | 仅训练 |
| `VideoRandomGrayscale` | 随机灰度化 | 仅训练 |
| `VideoGrayscale` | 随机灰度化 | 仅训练 |
| `VideoRandomPosterize` | 随机色调分离 | 仅训练 |
| `VideoHorizontalFlip` | 随机水平翻转 | 仅训练 |
| `VideoRandomRotation` | 随机旋转 | 仅训练 |

### 8. `VideoFocusRect`

为 EgoView（第一人称视角）设计的聚焦变换。对视频中指定矩形区域外的区域施加模糊或噪声效果：

```python
class VideoFocusRect(ModalityTransform):
    xtl, ytl: float  # 矩形左上角（归一化坐标）
    xbr, ybr: float  # 矩形右下角（归一化坐标）
    jitter: float     # 坐标抖动量
    blur_kernel: int  # 模糊核大小
    noise_std: float  # 噪声标准差
```

工作流程：
1. 检测图像中的 padding 区域
2. 在内容区域内定义聚焦矩形
3. 创建平滑混合 mask（使用距离变换）
4. 对矩形外区域施加模糊或噪声

### 9. `VideoNormalize`

对已转为浮点张量的视频进行均值/标准差归一化：

```python
class VideoNormalize(VideoTransform):
    mean: list[float]  # 各通道均值
    std: list[float]   # 各通道标准差
```

### 10. 已弃用的类

`VideoMergeTimeBatch` 和 `VideoSplitTimeBatch` 已弃用，功能由 `ComposedModalityTransform` 取代。

## 核心类/函数表格

| 名称 | 类型 | 说明 |
|------|------|------|
| `VideoTransform` | 抽象基类 | 所有视频变换的基类，支持双后端 |
| `VideoCrop` | 具体变换 | 训练随机裁剪 / 评估中心裁剪 |
| `VideoResize` | 具体变换 | 分辨率缩放 |
| `VideoToTensor` | 具体变换 | numpy uint8 -> torch float32 |
| `VideoToNumpy` | 具体变换 | torch float32 -> numpy uint8 |
| `VideoRandomErasing` | 具体变换 | 随机遮挡 |
| `VideoColorJitter` | 具体变换 | 颜色抖动 |
| `VideoFocusRect` | 具体变换 | 聚焦矩形（EgoView 专用） |
| `VideoNormalize` | 具体变换 | 均值/标准差归一化 |

## 与其他模块的关系

- **继承 `base.py`** - 继承 `ModalityTransform`
- **依赖 `schema`** - 使用 `DatasetMetadata` 获取视频分辨率
- **依赖 torchvision / albumentations** - 实际的图像变换操作
- **依赖 einops** - 多维张量重排
- **依赖 OpenCV** - `VideoFocusRect` 使用 cv2 进行模糊和距离变换

## 总结

`video.py` 提供了全面的视频数据预处理能力，涵盖了从格式转换到数据增强的完整流程。双后端设计（torchvision/albumentations）增加了灵活性，`ReplayCompose` 机制保证了同一视频中不同帧应用相同的随机增强参数。`VideoFocusRect` 是专门为机器人第一人称视角设计的创新增强方法，通过模糊/噪声化背景来引导模型关注操作区域。
