# `utils.py` — 训练工具函数集

## 文件概述

`utils.py` 提供了一组服务于训练流程的工具函数，涵盖数据类型转换、分布式打印、checkpoint 路径检测、模型保存、梯度累积计算和模型参数信息统计。这些函数被 `base.py` 和 `experiment.py` 广泛调用，是实验模块的基础设施层。

## 关键代码解析

### 1. dtype_from_string — 数据类型转换

```python
def dtype_from_string(dtype_str):
    if dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype_str {dtype_str}")
```

将配置文件中的字符串（如 `"bfloat16"`）转换为 PyTorch 的 dtype 对象。这是因为 Hydra/OmegaConf 配置只能表示基本数据类型，无法直接存储 `torch.bfloat16` 这样的 Python 对象。

### 2. rprint / mprint — 分布式打印函数

```python
def rprint(*args, **kwargs):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        return print(f"[dist-{rank}-of-{world_size}]", *args, **kwargs)
    else:
        return print(*args, **kwargs)

def mprint(*args, **kwargs):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        if rank == 0:
            return print(f"[dist-{rank}-of-{world_size}]", *args, **kwargs)
        else:
            return
    else:
        return print(*args, **kwargs)
```

两个函数的区别：
- **`rprint`** (rank print)：所有进程都打印，输出带有 rank 前缀（如 `[dist-3-of-8]`），用于调试分布式问题。
- **`mprint`** (main print)：仅主进程（rank 0）打印，非主进程静默返回，用于避免多进程重复输出。

### 3. get_checkpoint_path — 检查点路径检测

```python
def get_checkpoint_path(output_dir: str, checkpoint_prefix: str = "checkpoint") -> str | None:
    output_dir = os.path.abspath(output_dir)
    pathlib_dir = pathlib.Path(output_dir)

    if list(pathlib_dir.glob("config.json")):
        # 训练已完成
        return output_dir, False
    else:
        try:
            ordering_and_checkpoint_path = []
            glob_checkpoints = [
                str(x)
                for x in pathlib.Path(output_dir).glob(f"{checkpoint_prefix}-*")
                if os.path.isdir(x)
            ]
            for path in glob_checkpoints:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))
            checkpoints_sorted = sorted(ordering_and_checkpoint_path)
            return checkpoints_sorted[-1][1], True
        except IndexError:
            return None, True
```

该函数返回一个二元组 `(path, continue_training)`，用于判断训练状态：

| 情况 | 返回值 | 含义 |
|------|--------|------|
| 存在 `config.json` | `(output_dir, False)` | 训练已完成，无需继续 |
| 存在 `checkpoint-*` 目录 | `(最新checkpoint路径, True)` | 需要从最新 checkpoint 恢复 |
| 目录为空或无 checkpoint | `(None, True)` | 首次训练，从头开始 |

检测逻辑：
1. 先检查输出目录是否存在 `config.json`。该文件由 `model.save_pretrained()` 在训练完成后生成，因此其存在表明训练已结束。
2. 否则，搜索所有 `checkpoint-{step}` 目录，提取步数排序，返回最新的 checkpoint 路径。
3. 如果没有找到任何 checkpoint，返回 `None` 表示需要从头训练。

### 4. safe_save_model_for_hf_trainer — 安全保存模型

```python
def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir, _internal_call=True)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
```

**两种保存路径**：
- **DeepSpeed 模式**：先同步 CUDA 操作，然后调用 `trainer.save_model`。DeepSpeed 有自己的分片保存机制，需要通过其 API 收集分布在不同进程上的模型参数。
- **非 DeepSpeed 模式**：将整个 state dict 移动到 CPU 后保存。先 `del state_dict` 释放 GPU 上的副本，减少峰值显存占用。`should_save` 确保只有指定的进程（通常是 rank 0）执行实际保存操作。

### 5. compute_grad_accum_to_match_global_bs — 梯度累积步数计算

```python
def compute_grad_accum_to_match_global_bs(global_bs: int, bs: int):
    num_devices = torch.distributed.get_world_size()
    per_step_bs = bs * num_devices
    assert global_bs % per_step_bs == 0, f"{global_bs=}, {per_step_bs=}"
    num_grad_accum = global_bs // per_step_bs
    return num_grad_accum
```

根据目标全局 batch size 和当前的设备数/单卡 batch size 自动计算梯度累积步数。公式：

```
梯度累积步数 = global_batch_size / (per_device_batch_size * num_devices)
```

这确保无论使用多少 GPU，每次优化器更新都基于相同数量的样本，保证实验的可复现性。断言确保整除，避免 batch size 配置错误。

### 6. get_training_param_info — 训练参数状态树

```python
def get_training_param_info(model):
    module_states = dict()
    for module_name, module in model.named_children():
        key = f"{module_name}({module.__class__.__name__})"
        if all([p.requires_grad for p in module.parameters()]):
            module_states[key] = "true"
        elif all([not p.requires_grad for p in module.parameters()]):
            module_states[key] = "false"
        else:
            module_states[key] = get_training_param_info(module)
    return module_states
```

递归遍历模型的子模块，生成一个字典树，标注每个模块的梯度状态：
- `"true"`：模块所有参数都需要梯度（全量训练）。
- `"false"`：模块所有参数都冻结。
- 嵌套字典：部分参数需要梯度（如 LoRA 微调时，只有 LoRA adapter 参数需要梯度）。

这对于调试参数冻结策略非常有用，可以快速查看哪些模块在训练。

### 7. get_param_count_tree — 参数量统计树

```python
def get_param_count_tree(model: nn.Module):
    def format_param_count(count: int) -> str:
        count_in_millions = count / 1e6
        if count_in_millions.is_integer():
            return f"{int(count_in_millions)}M"
        else:
            return f"{count_in_millions:.2f}M"

    def module_to_dict(module: nn.Module, module_name: str) -> dict:
        total_count = sum(p.numel() for p in module.parameters())
        formatted_total = format_param_count(total_count)
        key = f"{module_name} ({module.__class__.__name__}, {formatted_total})"

        children = list(module.named_children())
        if children:
            nested = {}
            for child_name, child_module in children:
                nested.update(module_to_dict(child_module, child_name))
            return {key: nested}
        else:
            return {key: {}}

    nested_dict = module_to_dict(model, "model")
    return nested_dict
```

递归生成模型的参数量统计树。输出示例：

```json
{
  "model (VLAModel, 7800.50M)": {
    "backbone (VisionLanguageModel, 7500M)": {
      "vision_tower (CLIPVisionTower, 300M)": {},
      "language_model (LlamaForCausalLM, 7200M)": {}
    },
    "action_head (ActionHead, 300.50M)": {}
  }
}
```

参数量以百万（M）为单位显示，整数时省略小数点。叶子模块的值为空字典。

## 核心类/函数表格

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `dtype_from_string` | `dtype_str: str` | `torch.dtype` | 字符串转 PyTorch dtype |
| `rprint` | `*args, **kwargs` | — | 所有 rank 打印（带 rank 前缀） |
| `mprint` | `*args, **kwargs` | — | 仅主进程打印 |
| `is_local` | `model_name_or_path: str` | `bool` | 判断是否为本地路径 |
| `get_checkpoint_path` | `output_dir, checkpoint_prefix` | `(path, bool)` | 检测最新 checkpoint 和训练状态 |
| `prepare_config_for_training` | `config, model_args, training_args, data_args` | `None` | 设置模型配置的训练参数 |
| `safe_save_model_for_hf_trainer` | `trainer, output_dir` | — | 安全保存模型到磁盘 |
| `compute_grad_accum_to_match_global_bs` | `global_bs, bs` | `int` | 计算梯度累积步数 |
| `get_training_param_info` | `model` | `dict` | 获取模块参数训练状态树 |
| `get_param_count_tree` | `model` | `dict` | 获取模块参数量统计树 |

## 与其他模块的关系

- **被 `base.py` 导入**：`compute_grad_accum_to_match_global_bs`、`dtype_from_string`、`get_checkpoint_path`、`mprint`、`safe_save_model_for_hf_trainer` 均在 `BaseExperiment` 和 `BaseTrainer` 中使用。
- **`transformers.Trainer`**：`safe_save_model_for_hf_trainer` 直接操作 Trainer 实例。
- **`transformers.PretrainedConfig`**：`prepare_config_for_training` 设置配置对象的训练属性。
- **`torch.distributed`**：`compute_grad_accum_to_match_global_bs` 调用 `get_world_size()` 获取设备数。

## 总结

`utils.py` 是实验模块的工具库，提供了训练框架所需的各种辅助功能。这些函数的设计理念是将分布式训练中的常见操作（打印控制、checkpoint 管理、batch size 计算、模型保存）封装为可复用的原子操作。特别值得关注的是 `get_checkpoint_path` 的三态检测逻辑和 `compute_grad_accum_to_match_global_bs` 的全局 batch size 保证机制，它们共同确保了训练过程的健壮性和可复现性。
