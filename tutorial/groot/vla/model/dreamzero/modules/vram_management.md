# `vram_management.py` -- GPU 显存管理模块

## 文件概述

本文件实现了 DreamZero 的 GPU 显存（VRAM）管理机制，通过将模型权重存储在 CPU 上并在计算时按需加载到 GPU，使得 14B 参数的 Wan2.1 DiT 模型可以在有限显存的单 GPU 上运行。同时支持 FP8 量化推理和 LoRA 权重融合。

## 关键代码解析

### AutoWrappedLinear -- 自动管理的线性层

```python
class AutoWrappedLinear(torch.nn.Linear, AutoTorchModule):
    def __init__(self, module, offload_dtype, offload_device,
                 onload_dtype, onload_device,
                 computation_dtype, computation_device, vram_limit, ...):
        self.weight = module.weight  # 共享原始权重
        self.state = 0  # 0=offloaded, 1=onloaded, 2=kept
        self.enable_fp8 = computation_dtype in [torch.float8_e4m3fn, ...]

    def forward(self, x, *args, **kwargs):
        # VRAM 管理：按需加载权重
        if self.state == 2:
            weight, bias = self.weight, self.bias
        elif self.onload_dtype == self.computation_dtype:
            weight, bias = self.weight, self.bias
        elif self.vram_limit is not None and self.check_free_vram():
            self.keep()  # 显存充足，保持在 GPU
            weight, bias = self.weight, self.bias
        else:
            weight = cast_to(self.weight, self.computation_dtype, self.computation_device)
            bias = cast_to(self.bias, ...) if self.bias else None

        # FP8 推理或标准推理
        if self.enable_fp8:
            out = self.fp8_linear(x, weight, bias)
        else:
            out = torch.nn.functional.linear(x, weight, bias)

        # LoRA 推理
        if len(self.lora_A_weights) > 0:
            for lora_A, lora_B in zip(self.lora_A_weights, self.lora_B_weights):
                out = out + x @ lora_A.T @ lora_B.T
        return out
```

关键设计：
1. **三态管理**: offloaded (CPU) -> onloaded (GPU, 可能不同 dtype) -> kept (GPU, 计算 dtype)
2. **动态 VRAM 检查**: 通过 `check_free_vram()` 判断是否有足够显存保持权重
3. **FP8 量化**: 支持 `float8_e4m3fn` 和 `float8_e4m3fnuz` 量化推理
4. **LoRA 融合**: 内置 LoRA A/B 权重的高效推理

### FP8 线性计算

```python
def fp8_linear(self, input, weight, bias=None):
    x_max = torch.max(torch.abs(input), dim=-1, keepdim=True).values
    fp8_max = 448.0
    scale_a = torch.clamp(x_max / fp8_max, min=1.0)
    input = (input / (scale_a + 1e-8)).to(self.computation_dtype)
    weight = weight.to(self.computation_dtype)
    result = torch._scaled_mm(input, weight.T, scale_a=scale_a, scale_b=scale_b.T, bias=bias)
    return result
```

使用 `torch._scaled_mm` 进行 FP8 矩阵乘法，通过动态缩放避免数值溢出。

### enable_vram_management -- 递归替换

```python
def enable_vram_management(model, module_map, module_config, ...):
    enable_vram_management_recursively(model, module_map, module_config, ...)

def enable_vram_management_recursively(model, module_map, module_config, ...):
    for name, module in model.named_children():
        for source_module, target_module in module_map.items():
            if isinstance(module, source_module):
                module_ = target_module(module, **module_config)
                setattr(model, name, module_)
                break
        else:
            enable_vram_management_recursively(module, ...)
```

递归遍历模型的所有子模块，将 `nn.Linear` 替换为 `AutoWrappedLinear`，`nn.LayerNorm` 替换为 `WanAutoCastLayerNorm`。

## 核心类/函数表格

| 类/函数名 | 功能描述 |
|-----------|---------|
| `AutoTorchModule` | 自动管理基类，提供 offload/onload/keep 方法 |
| `AutoWrappedModule` | 通用模块包装器 |
| `AutoWrappedLinear` | 线性层包装器（含 FP8 + LoRA） |
| `WanAutoCastLayerNorm` | LayerNorm 包装器 |
| `enable_vram_management()` | 启用模型的 VRAM 管理 |
| `cast_to()` | 数据类型和设备转换 |

## 与其他模块的关系

- **被 `wan_flow_matching_action_tf.py` 调用**: `enable_vram_management()` 在推理初始化时调用
- **使用 `utils.py`**: `init_weights_on_device` 用于 meta 设备初始化

## 总结

VRAM 管理模块是 DreamZero 在有限硬件资源下部署大模型的关键技术。通过将权重按需在 CPU 和 GPU 之间搬运，配合 FP8 量化和 LoRA 融合，使得 14B 参数模型可以在 24GB 显存的消费级 GPU 上运行推理。
