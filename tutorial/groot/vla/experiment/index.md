# `groot/vla/experiment` 模块 — 实验训练框架总览

## 文件概述

`experiment` 模块是 DreamZero 项目中 VLA (Vision-Language-Action) 模型训练的核心框架层。它基于 HuggingFace `transformers.Trainer` 构建,提供了从实验配置、模型创建、数据加载到分布式训练的完整流程管理。

本模块包含以下文件:

| 文件 | 说明 |
|------|------|
| `__init__.py` | 模块入口,导出 `VLATrainer` |
| `base.py` | 基础训练器 `BaseTrainer`、基础实验 `BaseExperiment`、采样器及回调 |
| `experiment.py` | VLA 专用训练器 `VLATrainer` 和实验类 `VLAExperiment`,以及训练入口 `main()` |
| `utils.py` | 工具函数:dtype 转换、checkpoint 路径解析、分布式打印、模型保存等 |

## 模块架构图

```
+---------------------------------------------------------------+
|                       experiment 模块                          |
+---------------------------------------------------------------+
|                                                               |
|  utils.py                                                     |
|  +-----------------------------------------------------------+|
|  | dtype_from_string()  mprint() / rprint()                  ||
|  | get_checkpoint_path()  safe_save_model_for_hf_trainer()   ||
|  | compute_grad_accum_to_match_global_bs()                   ||
|  | get_training_param_info()  get_param_count_tree()         ||
|  +-----------------------------------------------------------+|
|            |                    |                    |         |
|            v                    v                    v         |
|  base.py                                                      |
|  +-----------------------------------------------------------+|
|  |                                                           ||
|  |  Callbacks:                                               ||
|  |  +------------------+ +---------------------+            ||
|  |  |LossLoggerCallback| |CheckpointFormatCB   |            ||
|  |  +------------------+ +---------------------+            ||
|  |  +------------------+                                     ||
|  |  |ProfCallback      |  (PyTorch Profiler 管理)            ||
|  |  +------------------+                                     ||
|  |                                                           ||
|  |  +------------------+                                     ||
|  |  |BaseSampler       |  (支持 set_epoch 的采样器)           ||
|  |  +------------------+                                     ||
|  |                                                           ||
|  |  +------------------+     继承自                           ||
|  |  |BaseTrainer       | <-- transformers.Trainer             ||
|  |  +------------------+                                     ||
|  |         |                                                  ||
|  |  +------------------+                                     ||
|  |  |BaseExperiment    |  (ABC, 实验编排基类)                  ||
|  |  +------------------+                                     ||
|  +-----------------------------------------------------------+|
|            |                                                   |
|            v                                                   |
|  experiment.py                                                |
|  +-----------------------------------------------------------+|
|  |  +------------------+     继承自                           ||
|  |  |VLATrainer        | <-- BaseTrainer                      ||
|  |  +------------------+                                     ||
|  |  +------------------+                                     ||
|  |  |VLATrainerInfer.. | <-- VLATrainer (推理基准测试)        ||
|  |  +------------------+                                     ||
|  |  +------------------+     继承自                           ||
|  |  |VLAExperiment     | <-- BaseExperiment                   ||
|  |  +------------------+                                     ||
|  |                                                           ||
|  |  main()  <-- @hydra.main  (训练入口点)                     ||
|  +-----------------------------------------------------------+|
|                                                               |
+---------------------------------------------------------------+
                            |
           外部依赖:        |
    +-----------------------+--------------------------+
    |                       |                          |
    v                       v                          v
transformers.Trainer   hydra/OmegaConf          groot.vla.data.*
                                               groot.vla.utils.*
```

## 调用流程

```
main() (experiment.py)
  |
  +-> apply_action_overrides(cfg)     # 覆盖 action 配置
  +-> VLAExperiment(cfg)              # 创建实验实例
  |     |
  |     +-> BaseExperiment.__init__()
  |     |     +-> 校验配置 (transforms, max_steps 等)
  |     |     +-> instantiate(cfg.training_args)
  |     |     +-> 配置 wandb 环境变量
  |     |     +-> get_checkpoint_path()   # 检查是否断点续训
  |     |     +-> create_model()          # 实例化模型,加载预训练权重
  |     |     +-> create_train_dataset()  # 创建训练数据集
  |     |     +-> create_trainer()        # 创建 VLATrainer
  |     |           +-> 计算梯度累积步数
  |     |           +-> 注册 Callbacks
  |     |
  |     +-> 保存 initial_actions (仅真实机器人数据集)
  |
  +-> experiment.train()
        +-> trainer.train(resume_from_checkpoint=...)
        +-> safe_save_model_for_hf_trainer()
```

## 设计要点

1. **分层继承**: `BaseTrainer -> VLATrainer`, `BaseExperiment -> VLAExperiment`,便于扩展到其他实验类型。
2. **配置驱动**: 深度依赖 Hydra/OmegaConf,所有组件通过 `instantiate()` 动态创建,实现配置与代码解耦。
3. **分布式友好**: 全局 batch size 自动计算梯度累积步数;采样器支持 `set_epoch` 保证可复现性;打印函数区分 rank。
4. **断点续训**: 自动检测已有 checkpoint 并恢复训练状态,同时处理 `ShardedLeRobotMixtureDataset` 的种子重置。
5. **可观测性**: 内置 Loss 日志 (JSONL)、PyTorch Profiler 回调、WandB 集成。
