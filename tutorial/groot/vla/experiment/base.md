# `base.py` — 训练框架基础组件

## 文件概述

`base.py` 是 DreamZero VLA 训练框架的核心基础文件。它定义了：

- **三个训练回调** (`LossLoggerCallback`, `CheckpointFormatCallback`, `ProfCallback`)，分别负责损失日志记录、检查点格式化和 PyTorch 性能分析。
- **采样器** (`BaseSampler`)，为分布式训练提供可控的数据采样。
- **基础训练器** (`BaseTrainer`)，继承自 `transformers.Trainer`，定制了损失计算、优化器创建、模型保存等核心逻辑。
- **基础实验类** (`BaseExperiment`)，编排整个训练实验的生命周期：配置校验、模型创建、数据集创建、训练器创建和训练启动。

## 关键代码解析

### 1. NumPy 序列化修复与 LayerNorm 层定义

文件开头有两段重要的全局设置：

```python
# Fix resume: https://github.com/huggingface/transformers/pull/34632/files
np_core = np.core
allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
allowlist += [type(np.dtype(np.uint32))]
torch.serialization.add_safe_globals(allowlist)
```

这段代码解决了 PyTorch 在加载 checkpoint 时对 NumPy 对象的反序列化安全限制问题。当 checkpoint 中包含 NumPy 数组（如 `TrainerState` 中的随机数状态）时，需要将这些类型加入安全白名单。

```python
LAYERNORM_LAYERS = [
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    # ... 其他归一化层
]
```

由于 `transformers` 库移除了已废弃的 `ALL_LAYERNORM_LAYERS`，这里本地定义了所有归一化层类型列表，用于优化器中区分需要 weight decay 的参数。

### 2. LossLoggerCallback — 损失日志回调

```python
class LossLoggerCallback(TrainerCallback):
    def __init__(self, output_path: str):
        self.output_path = output_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        entry = {"step": state.global_step}
        for key in ("loss", "dynamics_loss_avg", "action_loss_avg", "learning_rate"):
            if key in logs:
                entry[key] = logs[key]
        if len(entry) > 1:
            with open(self.output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
```

**核心逻辑**：
- 仅在主进程（`is_world_process_zero`）上写入日志，避免多进程重复写入。
- 将每步的 `loss`、`dynamics_loss_avg`、`action_loss_avg`、`learning_rate` 追加写入 JSONL 文件。
- JSONL 格式便于后续离线分析和可视化。

### 3. CheckpointFormatCallback — 检查点格式化回调

```python
class CheckpointFormatCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            # 复制实验配置目录
            if self.exp_cfg_dir is not None:
                shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)
            # 复制 processor 目录
            if self.processor_dir is not None:
                shutil.copytree(self.processor_dir, checkpoint_dir, dirs_exist_ok=True)
            # 复制 wandb_config.json
            ...
```

**设计意图**：让每个 checkpoint 都是"独立"的（self-contained），将实验配置文件（`conf.yaml`、`metadata.json`、`initial_actions.npz`）和 WandB 配置复制到每个 checkpoint 目录下，方便后续直接从任意 checkpoint 恢复评估而无需额外的配置文件。

### 4. ProfCallback — PyTorch Profiler 回调

```python
class ProfCallback(transformers.TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        # 在指定的 session step 窗口内启动 profiler
        if self.session_step == self.profile_start_step and self.prof is None:
            self.prof = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    skip_first=0, wait=0,
                    warmup=self.warmup_steps,
                    active=self.active_steps, repeat=1,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.profile_dir)),
            )
            self.prof.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        if self.profiling_active and self.prof is not None:
            self.prof.step()
        # 到达结束窗口时停止 profiler 并释放资源
        if self.session_step == self.profile_end_step and self.prof is not None:
            self.prof.__exit__(None, None, None)
            del self.prof
            self.prof = None
            torch.cuda.synchronize()
            self.profiling_complete = True
```

**核心特性**：
- **动态窗口控制**：通过 `profile_start_step` 和 `profile_end_step` 指定分析窗口，避免全程分析的巨大开销。
- **资源释放**：分析完成后显式删除 profiler 对象并同步 CUDA，确保 CUPTI 资源被释放。
- **自我移除**：分析完成并上传后，回调从 trainer 中移除自身，消除后续训练步骤的回调开销。
- **Trace 输出**：生成 TensorBoard 兼容的 trace 文件，可用于性能分析可视化。

### 5. BaseSampler — 自定义采样器

```python
class BaseSampler(Sampler):
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            self.data_source.set_epoch(epoch)
```

**设计要点**：
- `set_epoch` 方法由 HuggingFace Trainer 在每个 epoch 结束时自动调用，向下传递给数据集。
- 随机排列使用 `self.seed + self.epoch` 作为种子，保证跨 epoch 的随机性变化，同时保证可复现性。
- 注意注释中强调"不能加 rank"，确保所有进程看到相同的采样顺序（分布式训练中的数据分片由其他机制处理）。

### 6. BaseTrainer — 基础训练器

#### 初始化

```python
class BaseTrainer(transformers.Trainer):
    def __init__(self, **kwargs):
        torch._dynamo.config.cache_size_limit = 1000
        self.compute_dtype = kwargs.pop("compute_dtype")
        self.output_dir = kwargs.pop("output_dir")
        self.timer = ContextTimer(self)
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        # ... 分布式环境变量读取
        # ... 性能分析配置
        super().__init__(**kwargs)
        self.loss_queues = {}
        self.loss_queue_size = 10
```

`kwargs.pop()` 模式用于从传入参数中提取自定义参数，防止传递给父类 `Trainer.__init__` 时引发未知参数错误。`torch._dynamo.config.cache_size_limit = 1000` 增大了 TorchDynamo 缓存，以适应不同帧数的视频输入。

#### 损失计算与移动平均

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    with self.timer.with_label("model_forward"):
        outputs = model(inputs)
    for key, value in outputs.items():
        if key.endswith("_loss") and key != "loss":
            if key not in self.loss_queues:
                self.loss_queues[key] = []
            current_value = value.item() if torch.is_tensor(value) else value
            self.loss_queues[key].append(current_value)
            if len(self.loss_queues[key]) > self.loss_queue_size:
                self.loss_queues[key].pop(0)
            if self.current_step % self.loss_queue_size == 0:
                avg_loss = sum(self.loss_queues[key]) / len(self.loss_queues[key])
                self.log({f"{key}_avg": avg_loss})
    loss = outputs["loss"]
    return (loss, outputs) if return_outputs else loss
```

该方法覆盖了 `Trainer.compute_loss`，自动追踪模型输出中所有以 `_loss` 结尾的附加损失项，并通过滑动窗口（大小 10）计算移动平均值进行日志记录。这对于同时训练 dynamics loss 和 action loss 等多目标任务非常有用。

#### 自定义优化器

```python
def create_optimizer(self):
    decay_parameters = get_parameter_names(opt_model, LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {"params": [...需要 weight_decay 的参数...], "weight_decay": self.args.weight_decay},
        {"params": [...不需要 weight_decay 的参数...], "weight_decay": 0.0},
    ]
```

将参数分为两组：归一化层和偏置项不施加 weight decay，其余参数正常施加。这是深度学习训练中的标准做法，避免对不应正则化的参数施加惩罚。

#### 模型保存

```python
def save_model(self, output_dir, _internal_call):
    if self.base_cfg.save_lora_only:
        train_key = [k for k, v in self.model.named_parameters() if v.requires_grad]
        lora_state_dict = {k: v for k, v in self.model.state_dict().items() if k in train_key}
        state_dict = lora_state_dict
    if self.args.should_save:
        self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.base_cfg.save_llm:
            self.model.backbone.model.save_pretrained(os.path.join(output_dir, "llm"))
        if self.base_cfg.save_value_model:
            self.model.action_head.value_model.save_pretrained(...)
```

支持三种保存模式：
- **仅保存 LoRA 权重**（`save_lora_only`）：只保存 `requires_grad=True` 的参数，大幅减小 checkpoint 体积。
- **保存 LLM 子模型**（`save_llm`）：将 VLM 的语言模型骨干单独保存到 `llm/` 子目录，便于下游评估。
- **保存 Value Model**（`save_value_model`）：将 action head 中的价值模型单独保存。

#### 训练数据加载器（断点续训）

```python
def get_train_dataloader(self) -> DataLoader:
    if not isinstance(train_dataset, (ShardedLeRobotMixtureDataset)):
        return super().get_train_dataloader()
    self.args.ignore_data_skip = True
    curr_global_step = self.state.global_step
    if curr_global_step > 0:
        new_seed = train_dataset.seed + curr_global_step
        train_dataset.reset_seed(new_seed)
```

对于 `ShardedLeRobotMixtureDataset`，不使用 HuggingFace 默认的数据跳过策略（`ignore_data_skip = True`），而是通过重置随机种子来实现"伪断点续训"。注释说明这会使实验不完全可复现，但避免了大规模数据跳过的开销。

### 7. BaseExperiment — 实验编排基类

```python
class BaseExperiment(ABC):
    def __init__(self, cfg: DictConfig):
        # 1. 配置校验
        assert cfg.max_steps > 0
        assert cfg.save_total_limit >= 5

        # 2. transforms 校验
        for tag, transform_cfg in cfg.transforms.items():
            _ = EmbodimentTag(tag)
            transform = instantiate(transform_cfg)
            assert isinstance(transform, ComposedModalityTransform)

        # 3. 训练参数实例化
        training_args = instantiate(cfg.training_args)
        set_seed(training_args.seed)

        # 4. WandB 环境变量配置
        # 5. 断点续训检测
        resume_path, continue_training = get_checkpoint_path(training_args.output_dir)

        # 6. 模型创建
        model = self.create_model(cfg, training_args)

        # 7. 数据集创建 + 元数据保存
        train_dataset = self.create_train_dataset(cfg, model)
        U.json_dump({...}, metadata_save_path)

        # 8. 训练器创建
        trainer = self.create_trainer(...)
```

**`create_model` 方法**支持两种预训练权重加载方式：
- **分片 safetensors**（`model.safetensors.index.json`）：逐分片加载以节省内存。
- **单文件 safetensors**（`model.safetensors`）：一次性加载。

加载后还可以触发延迟 LoRA 注入：`model.action_head.inject_lora_after_loading()`。

**`create_trainer` 方法**负责：
1. 根据 `global_batch_size` 自动计算梯度累积步数。
2. 使用 Hydra `instantiate` 以 partial 模式创建 trainer。
3. 注册所有回调（`CheckpointFormatCallback`、`LossLoggerCallback`、`ProfCallback`）。

## 核心类/函数表格

| 类/函数 | 类型 | 说明 |
|---------|------|------|
| `LossLoggerCallback` | `TrainerCallback` | 将每步损失指标写入 JSONL 文件 |
| `CheckpointFormatCallback` | `TrainerCallback` | 保存 checkpoint 时复制配置文件使其独立 |
| `ProfCallback` | `TrainerCallback` | 在指定步数窗口内运行 PyTorch Profiler |
| `BaseSampler` | `Sampler` | 支持 `set_epoch` 的数据采样器 |
| `BaseTrainer` | `transformers.Trainer` | 自定义损失计算、优化器、模型保存和数据加载 |
| `BaseExperiment` | `ABC` | 实验生命周期编排基类 |
| `LAYERNORM_LAYERS` | `list` | 所有归一化层类型列表，用于 weight decay 分组 |

## 与其他模块的关系

- **`experiment.utils`**：导入 `compute_grad_accum_to_match_global_bs`、`dtype_from_string`、`get_checkpoint_path`、`mprint`、`safe_save_model_for_hf_trainer`。
- **`groot.vla.data.dataset.lerobot_sharded`**：`ShardedLeRobotMixtureDataset` 用于自定义 DataLoader 逻辑。
- **`groot.vla.data.schema`**：`EmbodimentTag` 用于校验 transform 配置中的 tag。
- **`groot.vla.data.transform`**：`ComposedModalityTransform` 用于校验 transform 实例类型。
- **`groot.vla.utils.timer`**：`ContextTimer` 用于模型前向传播计时。
- **`groot.vla.common.utils`**：`json_dump` 用于保存元数据。
- **`transformers`**：`Trainer`、`TrainerCallback`、`TrainerState` 等核心训练基础设施。
- **`hydra` / `omegaconf`**：配置管理和组件实例化。

## 总结

`base.py` 是整个 VLA 训练框架的基石。它通过继承 `transformers.Trainer` 并重写关键方法（`compute_loss`、`create_optimizer`、`save_model`、`get_train_dataloader`），将 HuggingFace 的通用训练流程适配到 VLA 多模态训练场景。`BaseExperiment` 类则以模板方法模式编排了实验全流程，子类只需覆盖 `create_model`、`create_train_dataset` 等方法即可定制实验逻辑。三个回调类提供了训练过程的可观测性（损失日志、性能分析）和 checkpoint 的自包含性。
