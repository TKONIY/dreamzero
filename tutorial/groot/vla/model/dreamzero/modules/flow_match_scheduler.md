# `flow_match_scheduler.py` -- Flow Matching 调度器

## 文件概述

本文件实现了 Flow Matching 的噪声调度器，用于控制扩散过程中噪声的添加和去除。Flow Matching 是一种替代 DDPM/DDIM 的扩散训练方法，直接学习从噪声到数据的最优传输路径（Optimal Transport），训练更稳定且推理更快。

## 关键代码解析

### FlowMatchScheduler -- 调度器

```python
class FlowMatchScheduler:
    def __init__(self, num_inference_steps=100, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003/1.002, ...):
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, ...):
        # 生成等间距的 sigma 序列
        self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        # 应用 shift 变换（偏移采样分布，使更多步数集中在高噪声区域）
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps
```

`shift` 参数控制噪声调度的偏移：较大的 shift 值使更多采样步骤集中在高噪声区域，有助于捕获全局结构。

### 去噪步骤

```python
    def step(self, model_output, timestep, sample, to_final=False):
        sigma = self.sigmas[timestep_id]
        sigma_ = self.sigmas[timestep_id + 1] if not to_final else 0
        # Flow Matching 的简单欧拉步: x_{t+1} = x_t + v * (sigma_{t+1} - sigma_t)
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
```

Flow Matching 的去噪公式极其简洁：模型输出是速度场 `v`，通过 `x + v * delta_sigma` 完成一步去噪。

### 加噪和目标

```python
    def add_noise(self, original_samples, noise, timestep):
        sigma = self.sigmas[timestep_id]
        # 线性插值: x_t = (1 - sigma) * x_0 + sigma * noise
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample, noise, timestep):
        # 目标是 noise - sample（速度场方向）
        target = noise - sample
        return target
```

### 训练权重

```python
    def training_weight(self, timestep):
        # BSMNTW 加权：中间时间步权重高，两端低
        y = torch.exp(-2 * ((x - N/2) / N) ** 2)
        return y * (N / y.sum())
```

使用 Bell-shaped Monotonic Time-step Weighting (BSMNTW)，中间时间步（噪声中等的步骤）获得更高的训练权重。

## 核心类/函数表格

| 方法名 | 功能描述 |
|--------|---------|
| `set_timesteps()` | 设置推理步数和 sigma 序列 |
| `step()` | 单步去噪（欧拉法） |
| `add_noise()` | 按 sigma 添加噪声 |
| `training_target()` | 计算训练目标（速度场） |
| `training_weight()` | 计算时间步训练权重 |
| `return_to_timestep()` | 从稳定样本反推模型输出 |

## 与其他模块的关系

- **被 `wan_flow_matching_action_tf.py` 使用**: 训练和推理时的噪声调度
- **与 `flow_unipc_multistep_scheduler.py` 互补**: 后者提供更高阶的去噪求解器

## 总结

`FlowMatchScheduler` 是 DreamZero 训练和推理的时间调度核心。通过 Flow Matching 的线性插值范式，它将噪声控制简化为简洁的线性代数操作，配合 BSMNTW 训练权重，使模型在不同噪声水平上获得均衡的训练信号。
