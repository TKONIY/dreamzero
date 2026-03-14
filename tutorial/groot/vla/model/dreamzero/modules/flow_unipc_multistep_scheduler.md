# `flow_unipc_multistep_scheduler.py` -- Flow UniPC 多步调度器

## 文件概述

本文件实现了适配 Flow Matching 的 UniPC (Unified Predictor-Corrector) 多步求解器。UniPC 是一种高阶 ODE 求解器，可以在较少的步数内实现高质量的去噪，相比简单的欧拉法能显著减少推理时间。

## 关键代码解析

### FlowUniPCMultistepScheduler

```python
class FlowUniPCMultistepScheduler(SchedulerMixin, ConfigMixin):
    def __init__(self, num_train_timesteps=1000, solver_order=2,
                 prediction_type="flow_prediction", shift=1.0, ...):
        # 默认二阶求解器（实际精度为三阶）
        self.predict_x0 = predict_x0
        self.model_outputs = [None] * solver_order  # 缓存历史模型输出
```

### 多步预测器 (Predictor)

```python
@torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)
def multistep_uni_p_bh_update(self, model_output, sample, order, step_index):
    # 利用当前和历史的模型输出，通过多步法预测下一步
    sigma_t, sigma_s0 = self.sigmas[step_index + 1], self.sigmas[step_index]
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    h = lambda_t - lambda_s0

    # 计算差分 D1s
    for i in range(1, order):
        D1s.append((mi - m0) / rk)

    # 求解 Runge-Kutta 系数
    rhos_p = torch.linalg.solve_ex(R[:-1, :-1], b[:-1])[0]

    # 高阶预测
    x_t = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0 - alpha_t * B_h * pred_res
```

### 校正器 (Corrector)

```python
@torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)
def multistep_uni_c_bh_update(self, this_model_output, last_sample, this_sample, order, step_index):
    # 使用预测器的输出作为初始值，通过校正器提高精度
    D1_t = model_t - m0
    x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
```

### step 方法

```python
def step(self, model_output, timestep, sample, step_index, return_dict=True):
    # 1. 转换模型输出（flow_prediction -> x0_pred）
    model_output_convert = self.convert_model_output(model_output, sample, step_index)

    # 2. 校正步（如果不是第一步）
    if use_corrector:
        sample = self.multistep_uni_c_bh_update(...)

    # 3. 预测步
    prev_sample = self.multistep_uni_p_bh_update(...)
    return SchedulerOutput(prev_sample=prev_sample)
```

## 核心类/函数表格

| 方法名 | 功能描述 |
|--------|---------|
| `set_timesteps()` | 设置推理时间步 |
| `convert_model_output()` | 将 flow 预测转换为 x0 或 epsilon |
| `multistep_uni_p_bh_update()` | UniPC 多步预测器 |
| `multistep_uni_c_bh_update()` | UniPC 校正器 |
| `step()` | 完整的一步去噪 |
| `add_noise()` | 添加噪声 |

## 与其他模块的关系

- **与 `flow_match_scheduler.py` 互补**: 提供更高阶的求解器
- **被 `wan_flow_matching_action_tf.py` 使用**: 可配置使用此调度器加速推理

## 总结

`FlowUniPCMultistepScheduler` 通过二阶（或更高阶）的预测-校正方法，在保持生成质量的同时大幅减少推理步数。使用 `@torch.compile` 装饰器进一步优化了计算性能。
