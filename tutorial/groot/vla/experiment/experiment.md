# `experiment.py` — VLA 实验与训练器实现

## 文件概述

`experiment.py` 是 DreamZero VLA 模型训练的具体实现文件，也是训练流程的入口点。它在 `base.py` 提供的基础框架之上，定义了：

- **`VLATrainer`**：继承自 `BaseTrainer`，添加了时间基准测试、强制重启和 action head 的 global step 同步。
- **`VLATrainerInferenceBenchmark`**：继承自 `VLATrainer`，专门用于推理速度基准测试。
- **`VLAExperiment`**：继承自 `BaseExperiment`，在实验初始化时额外保存初始动作数据。
- **`main()`**：Hydra 驱动的训练入口函数。

## 关键代码解析

### 1. ForceRestart 异常

```python
class ForceRestart(ValueError):
    pass
```

一个自定义异常类，当训练时间超过预设上限时抛出，用于触发训练作业的强制重启。这在长时间训练任务中非常有用，例如在集群调度系统中定期重启以释放资源或适应调度策略。

### 2. VLATrainer — VLA 训练器

```python
class VLATrainer(BaseTrainer):
    def __init__(self, **kwargs):
        self.benchmark_time = kwargs.pop("benchmark_time", False)
        self.step_timer = None
        self.num_trials = kwargs.pop("num_trials", 10)
        self.curr_trial = 0
        self.all_times = []
        self.start_time = time.time()
        self.restart_max_seconds = kwargs.pop("restart_max_seconds", 0)
        import torch.distributed as dist
        self.rank = dist.get_rank()
        self.micro_global_step = 0
        super().__init__(**kwargs)
```

`VLATrainer` 通过 `kwargs.pop()` 提取自己的参数（`benchmark_time`、`num_trials`、`restart_max_seconds`），然后将剩余参数传给 `BaseTrainer`。`micro_global_step` 用于追踪每次 forward 调用（包括梯度累积中的微步），区别于 `state.global_step`（仅在优化器更新时增加）。

#### training_step 方法

```python
def training_step(self, model, inputs, *args, **kwargs):
    self.micro_global_step += 1

    if hasattr(self.model.action_head, "global_step"):
        self.model.action_head.global_step = self.state.global_step

    if self.benchmark_time:
        if self.state.global_step % 100 == 0:
            if self.step_timer is not None:
                elapsed_time = time.time() - self.step_timer
                self.all_times.append(elapsed_time)
                self.curr_trial += 1
            self.step_timer = time.time()
        if self.curr_trial >= self.num_trials:
            exit(0)

    if self.state.global_step % self.state.save_steps == 1:
        if self.restart_max_seconds > 0:
            cur_time = time.time()
            if (cur_time - self.start_time) > self.restart_max_seconds:
                raise ForceRestart(f"Exceeded time limit {self.restart_max_seconds} seconds")

    loss_dict = super().training_step(model, inputs, *args, **kwargs)
    return loss_dict
```

**三个核心功能**：

1. **Action Head 步数同步**：将 trainer 的 `global_step` 同步到 `model.action_head.global_step`。某些 action head 实现（如带有学习率调度或渐进策略的 head）需要知道当前训练步数来调整行为。

2. **时间基准测试**：当 `benchmark_time=True` 时，每 100 步记录一次运行时间。收集 `num_trials` 次数据后自动退出，用于衡量训练吞吐量。

3. **强制重启机制**：在每个保存步（`save_steps`）之后的第一步检查是否超时。选择在 save 之后检查是因为此时模型状态已安全保存，重启不会丢失进度。

### 3. VLATrainerInferenceBenchmark — 推理基准测试

```python
class VLATrainerInferenceBenchmark(VLATrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        warmup_steps = 100
        measure_steps = 100

        model.eval()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.inference_mode():
                for i in range(warmup_steps):
                    action = model.module.get_action(inputs)
                    action.keys()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.inference_mode():
                for i in range(measure_steps):
                    action = model.module.get_action(inputs)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        time_per_step = elapsed_time / measure_steps
        exit()
```

该类通过覆盖 `compute_loss` 方法，将训练流程"劫持"为推理基准测试流程：

- 先执行 100 步预热，让 GPU 缓存和 CUDA kernel 稳定。
- 使用 `torch.cuda.Event` 精确测量 GPU 端的推理时间（比 `time.time()` 更准确，因为 CUDA 操作是异步的）。
- 测量 100 步后计算平均推理时间并退出。
- 使用 `torch.inference_mode()` 而非 `torch.no_grad()`，前者更激进地禁用自动微分以获得最佳推理性能。

### 4. VLAExperiment — VLA 实验类

```python
class VLAExperiment(BaseExperiment):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        if hasattr(self.train_dataset, "get_initial_actions"):
            initial_actions = self.train_dataset.get_initial_actions()
            if len(initial_actions) > 0:
                initial_actions_path = self.exp_cfg_dir / INITIAL_ACTIONS_FILENAME
                np.savez(str(initial_actions_path), initial_actions)
                print("Successfully dumped initial actions")
```

`VLAExperiment` 在 `BaseExperiment` 的初始化之上，额外处理初始动作的保存。初始动作数据结构如注释所述：

```
initial_actions: list[dict[str, dict[str, np.ndarray]]]
    0: (第一个数据集)
        trajectory_name:
            action_key:
                action: np.ndarray
    1: (第二个数据集)
        ...
```

初始动作仅在真实机器人数据集中存在（`get_initial_actions` 方法），模拟数据集则没有。保存为 `.npz` 格式便于后续策略部署时加载。

### 5. main() — Hydra 训练入口

```python
@hydra.main(config_path="../configs", config_name="conf", version_base=None)
def main(cfg):
    cfg = apply_action_overrides(cfg)
    experiment = VLAExperiment(cfg)
    experiment.train()
```

入口函数非常简洁：

1. Hydra 装饰器从 `groot/vla/configs/conf.yaml` 加载配置。
2. `apply_action_overrides` 根据配置中的覆盖规则更新 action 维度和 action horizon 参数。
3. 创建 `VLAExperiment` 实例（触发完整的初始化流程）。
4. 调用 `experiment.train()` 开始训练。

## 核心类/函数表格

| 类/函数 | 父类 | 说明 |
|---------|------|------|
| `ForceRestart` | `ValueError` | 训练超时强制重启异常 |
| `VLATrainer` | `BaseTrainer` | VLA 训练器,添加时间基准测试和重启机制 |
| `VLATrainerInferenceBenchmark` | `VLATrainer` | 推理速度基准测试训练器 |
| `VLAExperiment` | `BaseExperiment` | VLA 实验类,保存初始动作数据 |
| `main()` | — | Hydra 训练入口点 |
| `INITIAL_ACTIONS_FILENAME` | 常量 | 初始动作文件名 `"initial_actions.npz"` |

## 与其他模块的关系

- **`experiment.base`**：直接继承 `BaseExperiment` 和 `BaseTrainer`，是本文件的核心依赖。
- **`groot.vla.utils.action_args_override_utils`**：`apply_action_overrides` 用于动态调整 action 相关配置。
- **`groot.vla.configs/`**：Hydra 配置目录，`main()` 从此加载 `conf.yaml`。
- **`torch.distributed`**：`VLATrainer` 使用 `dist.get_rank()` 获取进程 rank。

## 总结

`experiment.py` 是 DreamZero VLA 训练的最终实现层。`VLATrainer` 在 `BaseTrainer` 的通用训练逻辑之上添加了 VLA 特有的功能：action head 状态同步、训练时间基准测试和超时重启保护。`VLATrainerInferenceBenchmark` 则将训练框架复用为推理性能测量工具。`VLAExperiment` 处理初始动作数据的持久化，而 `main()` 函数以极简的方式串联了配置加载、实验创建和训练启动的完整流程。
