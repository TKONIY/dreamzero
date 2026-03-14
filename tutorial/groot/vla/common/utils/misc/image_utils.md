# image_utils.py - 图像处理工具

## 文件概述

`image_utils.py` 提供了图像的格式转换、显示、读写以及预处理功能。该模块面向深度学习训练场景，特别处理了通道顺序（CHW vs HWC）、数据类型（uint8 vs float）、归一化等常见问题，并提供了基于 OpenCV 的实时图像显示类 `Cv2Display`。

## 关键代码解析

### 1. 图像格式转换 `to_image`

```python
def to_image(img, channel_order="auto"):
```

将各种格式的图像统一转换为 `[H, W, C]` 的 NumPy uint8 数组：
- 自动处理 PyTorch 张量（移到 CPU 并转换为 NumPy）
- 支持 4D 输入（自动去除 batch 维度，要求 batch=1）
- `"auto"` 模式通过检测 `shape[0] == 3` 来判断是 CHW 还是 HWC

### 2. 图像读写

```python
def imread(path, channel_order="chw", format="torch"):
def imsave(img, path):
```

- `imread` 使用 `imageio` 读取图像，默认输出 CHW 格式的 PyTorch 张量（适合直接输入神经网络）
- `imsave` 自动通过 `to_image` 转换后保存

### 3. `Cv2Display` 实时显示类

```python
class Cv2Display:
    def __init__(self, window_name="display", image_size=None, channel_order="auto",
                 bgr2rgb=True, step_sleep=0, enabled=True):
```

封装了 `cv2.imshow` 的实时图像显示窗口：
- 支持自动调整图像尺寸（根据目标尺寸选择合适的插值方法）
- 自动处理通道顺序转换
- 支持 BGR 到 RGB 的转换（OpenCV 默认使用 BGR）
- 处理 `DISPLAY` 环境变量，兼容 IsaacGym 等模拟器环境
- `enabled=False` 时所有操作为空操作

### 4. 图像张量健全性检查

```python
def sanity_check_image_tensor(img, on_error="raise"):
```

检查输入给神经网络的图像张量是否正确：
- 验证数据类型是否为浮点数（常见错误：忘记从 uint8 转换）
- 检查值是否都接近整数（常见错误：忘记归一化 `/255.0`）

这些检查可以及早发现训练中的数据预处理错误。

### 5. 基础图像预处理

```python
@torch.no_grad()
def basic_image_tensor_preprocess(img, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), shape=None):
```

执行标准的图像预处理流程：
1. 将输入展平到 4D（`[B, C, H, W]`）
2. 可选的尺寸调整（使用 `kornia.geometry.transform.resize`）
3. 除以 255 并进行均值/标准差归一化

使用 `global_once` 确保某些警告信息只打印一次。

## 核心类/函数表格

| 函数/类 | 说明 |
|---------|------|
| `to_image(img, channel_order)` | 将各种格式图像转为 HWC NumPy uint8 |
| `imshow(img)` | 使用 matplotlib 显示图像 |
| `imsave(img, path)` | 使用 imageio 保存图像 |
| `imread(path, channel_order, format)` | 读取图像为 CHW 格式张量 |
| `Cv2Display` 类 | 基于 OpenCV 的实时图像显示窗口 |
| `sanity_check_image_tensor(img)` | 检查图像张量的数据类型和归一化状态 |
| `basic_image_tensor_preprocess(img)` | 标准图像预处理（resize + normalize） |

## 与其他模块的关系

- **依赖** `array_tensor_utils.py`：使用 `any_describe` 进行形状检查的错误报告
- **依赖** `misc_utils.py`：使用 `global_once` 控制警告只打印一次
- **依赖** `torch_utils.py`：使用 `torch_normalize` 进行图像归一化
- **外部依赖**：`cv2`（OpenCV）、`imageio`、`matplotlib`、`kornia`
- **被视觉模型相关模块使用**：作为视觉输入预处理的标准工具

## 总结

`image_utils.py` 解决了深度学习图像处理中的几个常见痛点：通道顺序的混乱（CHW vs HWC）、数据类型的不一致（uint8 vs float）、以及忘记归一化等错误。`Cv2Display` 类特别适合机器人操作任务的实时可视化调试。`sanity_check_image_tensor` 作为防御性编程工具，能在训练早期就发现数据预处理错误，避免浪费计算资源。
