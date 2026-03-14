# DreamZero 代码教程：阅读指南

> 本指南帮助你根据自身目标选择最高效的代码阅读路径。DreamZero 代码库包含约 107 个源文件，本教程为每个文件提供逐行级别的中文解读。

---

## 目录

1. [三条阅读路径](#1-三条阅读路径)
2. [推荐模块阅读顺序](#2-推荐模块阅读顺序)
3. [完整文档索引](#3-完整文档索引)

---

## 1. 三条阅读路径

### 路径 A：快速入门（约 2 小时）

**适合人群**: 想要快速了解 DreamZero 是什么、能做什么、怎么跑起来的读者。

**阅读顺序**:

```
步骤 1 (20 分钟)  背景知识速览
  └─> 00_background_knowledge.md
       重点阅读：第 1-3 节（VLA 和世界模型概念）+ 第 7 节（架构图）

步骤 2 (20 分钟)  项目入口与配置
  └─> tutorial/root/pyproject.md            -- 项目依赖与安装
  └─> tutorial/groot/vla/configs/conf.md    -- 全局配置结构

步骤 3 (30 分钟)  核心模型架构
  └─> tutorial/groot/vla/model/dreamzero/base_vla.md
       -- VLA 基类：理解 Backbone + ActionHead 的组合方式
  └─> tutorial/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.md
       -- 动作头：Flow Matching 如何生成动作

步骤 4 (20 分钟)  推理流程
  └─> tutorial/root/socket_test_optimized_AR.md   -- 推理服务器
  └─> tutorial/root/test_client_AR.md              -- 测试客户端

步骤 5 (30 分钟)  训练流程
  └─> tutorial/scripts/train/droid_training.md     -- 训练脚本入口
  └─> tutorial/groot/vla/experiment/experiment.md  -- 训练循环逻辑
```

**阅读后你将能够**: 理解 DreamZero 的整体设计、运行推理服务器、启动训练任务。

---

### 路径 B：完整学习（约 2-3 天）

**适合人群**: 希望全面掌握代码库、准备修改或扩展 DreamZero 的开发者。

**阅读顺序**: 按照下方"推荐模块阅读顺序"（第 2 节）从头到尾阅读。每个模块附有预估时间。

**建议方式**:
- 第一天：背景知识 + 配置系统 + 数据管线 + 核心模型架构
- 第二天：DiT 模块 + 训练系统 + 推理系统
- 第三天：数据转换 + 评估工具 + 工具函数 + 查漏补缺

---

### 路径 C：按主题阅读

**适合人群**: 有明确目标的开发者，只想了解特定部分。

| 我想要... | 阅读这些文档 |
|-----------|-------------|
| **在新机器人上微调** | `00_background_knowledge.md` (第 7 节) --> `scripts/train/agibot_training.md` --> `scripts/train/yam_training.md` --> `scripts/data/convert_agibot.md` --> `groot/vla/data/schema/embodiment_tags.md` --> `groot/vla/configs/data/dreamzero/agibot_relative.md` |
| **理解 Flow Matching 实现** | `00_background_knowledge.md` (第 4-5 节) --> `groot/vla/model/dreamzero/modules/flow_match_scheduler.md` --> `groot/vla/model/dreamzero/modules/flow_unipc_multistep_scheduler.md` --> `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.md` |
| **理解视频生成** | `groot/vla/model/dreamzero/modules/wan_video_dit.md` --> `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.md` --> `groot/vla/model/dreamzero/modules/wan_video_vae.md` |
| **理解数据管线** | `groot/vla/data/dataset/lerobot.md` --> `groot/vla/data/dataset/lerobot_sharded.md` --> `groot/vla/data/transform/video.md` --> `groot/vla/data/transform/state_action.md` --> `groot/vla/data/transform/language.md` --> `groot/vla/data/transform/concat.md` |
| **部署推理服务** | `root/socket_test_optimized_AR.md` --> `root/test_client_AR.md` --> `groot/vla/model/dreamzero/modules/vram_management.md` --> `groot/vla/model/dreamzero/modules/attention.md` |
| **理解训练流程** | `groot/vla/experiment/experiment.md` --> `groot/vla/experiment/base.md` --> `scripts/train/droid_training.md` --> `groot/vla/configs/conf.md` |
| **转换自己的数据集** | `scripts/data/convert_droid.md` --> `scripts/data/convert_lerobot_to_gear.md` --> `scripts/data/convert_agibot.md` --> `groot/vla/data/dataset/metadata.md` |
| **理解注意力机制** | `groot/vla/model/dreamzero/modules/wan2_1_attention.md` --> `groot/vla/model/dreamzero/modules/attention.md` --> `groot/vla/model/dreamzero/modules/cudnn_attention.md` --> `groot/vla/model/dreamzero/modules/wan2_1_submodule.md` |
| **仿真评估** | `eval_utils/run_sim_eval.md` --> `eval_utils/policy_client.md` --> `eval_utils/policy_server.md` --> `groot/vla/model/n1_5/sim_policy.md` |

---

## 2. 推荐模块阅读顺序

以下是完整学习路径的推荐阅读顺序。顺序的设计原则是：**先建立全局理解，再深入局部细节；先理解数据流入，再理解模型处理，最后理解结果输出。**

### 第一阶段：基础与全局理解（约 2 小时）

| 顺序 | 模块 | 原因 |
|------|------|------|
| 1 | **背景知识** (`00_background_knowledge.md`) | 建立必要的领域知识基础 |
| 2 | **项目配置** (`root/pyproject.md`) | 了解依赖和项目结构 |
| 3 | **Hydra 配置系统** (`groot/vla/configs/`) | DreamZero 大量使用 Hydra 配置驱动，理解配置文件是读懂其他代码的前提 |

**为什么先读配置**: DreamZero 使用 Hydra 进行依赖注入。模型的构建、数据的加载、训练的参数全部由 YAML 配置驱动。不理解配置系统，代码中大量的 `instantiate()` 调用将无法理解。

### 第二阶段：数据管线（约 3 小时）

| 顺序 | 模块 | 原因 |
|------|------|------|
| 4 | **数据模式定义** (`groot/vla/data/schema/`) | 了解数据的结构约定，包括 embodiment 标签系统 |
| 5 | **数据集加载** (`groot/vla/data/dataset/`) | 理解 LeRobot 格式的数据如何被加载和索引 |
| 6 | **数据变换** (`groot/vla/data/transform/`) | 理解原始数据如何被转换为模型可接受的输入 |

**为什么先读数据**: "数据决定模型"——理解输入数据的格式和变换方式，才能理解模型为什么这样设计。

### 第三阶段：核心模型架构（约 4 小时）

| 顺序 | 模块 | 原因 |
|------|------|------|
| 7 | **VLA 基类** (`groot/vla/model/dreamzero/base_vla.py`) | 理解整体架构：Backbone + ActionHead 的组合 |
| 8 | **骨干网络** (`groot/vla/model/dreamzero/backbone/`) | 理解 Identity Backbone 的设计——DiT 本身即为骨干 |
| 9 | **编码器模块** (image_encoder, text_encoder) | 理解输入如何被编码为 DiT 的条件信号 |
| 10 | **DiT 核心** (`wan_video_dit.py`, `wan_video_dit_action_casual_chunk.py`) | 最核心的模块——视频和动作的联合去噪 Transformer |
| 11 | **注意力机制** (attention, wan2_1_attention, cudnn_attention) | DiT 的内部运算细节 |
| 12 | **VAE** (`wan_video_vae.py`) | 视频帧的编解码 |
| 13 | **Flow Matching** (flow_match_scheduler, flow_unipc_multistep_scheduler) | 去噪过程的调度策略 |
| 14 | **动作头** (`wan_flow_matching_action_tf.py`) | 将 DiT 输出转换为动作序列的最终环节 |
| 15 | **模型变换** (`groot/vla/model/dreamzero/transform/`) | 模型特有的数据预处理（区别于通用数据变换） |

**为什么 DiT 放在中间**: 先理解了输入编码（编码器）和输出解码（VAE、动作头），再深入 DiT 内部，可以带着"输入是什么、输出是什么"的清晰框架去理解中间的复杂逻辑。

### 第四阶段：训练与推理系统（约 2 小时）

| 顺序 | 模块 | 原因 |
|------|------|------|
| 16 | **训练实验框架** (`groot/vla/experiment/`) | 理解训练循环、损失计算、检查点保存 |
| 17 | **训练脚本** (`scripts/train/`) | 具体的启动命令和超参数配置 |
| 18 | **推理服务器** (`socket_test_optimized_AR.py`, `test_client_AR.py`) | 理解分布式推理和 DiT 缓存优化 |
| 19 | **显存管理** (`vram_management.py`) | 理解大模型推理时的显存优化技巧 |

### 第五阶段：数据转换与评估（约 1.5 小时）

| 顺序 | 模块 | 原因 |
|------|------|------|
| 20 | **数据转换脚本** (`scripts/data/`) | 如何将原始数据集转换为 DreamZero 所需格式 |
| 21 | **评估工具** (`eval_utils/`) | 仿真评估流程 |
| 22 | **N1.5 策略** (`groot/vla/model/n1_5/`) | 仿真评估中使用的策略包装器 |

### 第六阶段：工具与辅助（约 1 小时）

| 顺序 | 模块 | 原因 |
|------|------|------|
| 23 | **通用工具** (`groot/vla/common/utils/`) | 按需查阅即可，包含 IO、张量操作、视频处理等 |
| 24 | **辅助脚本** (`scripts/compare_loss.py`, `scripts/open_loop_yam.py`) | 调试和分析工具 |
| 25 | **VLA 工具** (`groot/vla/utils/`) | 动作参数覆盖、计时器等 |

---

## 3. 完整文档索引

### 基础文档

| 文档 | 说明 |
|------|------|
| [背景知识](00_background_knowledge.md) | VLA、世界模型、扩散模型、Flow Matching 等核心概念 |
| [阅读指南](00_reading_guide.md) | 本文档 |

### 项目根目录

| 文档 | 对应源文件 | 说明 |
|------|-----------|------|
| [socket_test_optimized_AR](root/socket_test_optimized_AR.md) | `socket_test_optimized_AR.py` | 分布式 WebSocket 推理服务器 |
| [test_client_AR](root/test_client_AR.md) | `test_client_AR.py` | 推理测试客户端 |
| [pyproject](root/pyproject.md) | `pyproject.toml` | 项目依赖与构建配置 |

### 评估工具 (eval_utils/)

| 文档 | 对应源文件 | 说明 |
|------|-----------|------|
| [policy_client](eval_utils/policy_client.md) | `eval_utils/policy_client.py` | 策略客户端，向推理服务器发送请求 |
| [policy_server](eval_utils/policy_server.md) | `eval_utils/policy_server.py` | 策略服务器封装 |
| [run_sim_eval](eval_utils/run_sim_eval.md) | `eval_utils/run_sim_eval.py` | 仿真评估启动脚本 |

### 训练脚本 (scripts/train/)

| 文档 | 对应源文件 | 说明 |
|------|-----------|------|
| [droid_training](scripts/train/droid_training.md) | `scripts/train/droid_training_lora.sh` | DROID LoRA 训练 (默认) |
| [droid_training_lora](scripts/train/droid_training_lora.md) | `scripts/train/droid_training_lora.sh` | DROID LoRA 微调训练 |
| [droid_training_full_finetune](scripts/train/droid_training_full_finetune.md) | `scripts/train/droid_training_full_finetune.sh` | DROID 全量微调训练 |
| [agibot_training](scripts/train/agibot_training.md) | `scripts/train/agibot_training.sh` | AgiBot 机器人训练 |
| [yam_training](scripts/train/yam_training.md) | `scripts/train/yam_training.sh` | YAM 机器人训练 |

### 数据转换脚本 (scripts/data/)

| 文档 | 对应源文件 | 说明 |
|------|-----------|------|
| [convert_droid](scripts/data/convert_droid.md) | `scripts/data/convert_droid.py` | DROID 数据集格式转换 |
| [convert_lerobot_to_gear](scripts/data/convert_lerobot_to_gear.md) | `scripts/data/convert_lerobot_to_gear.py` | LeRobot 到 GEAR 格式转换 |
| [convert_agibot](scripts/data/convert_agibot.md) | `scripts/data/convert_agibot.py` | AgiBot 数据集转换 |

### 其他脚本 (scripts/)

| 文档 | 对应源文件 | 说明 |
|------|-----------|------|
| [compare_loss](scripts/compare_loss.md) | `scripts/compare_loss.py` | 训练损失对比工具 |
| [open_loop_yam](scripts/open_loop_yam.md) | `scripts/open_loop_yam.py` | YAM 开环评估脚本 |

### 配置文件 (groot/vla/configs/)

| 文档 | 对应源文件 | 说明 |
|------|-----------|------|
| [conf](groot/vla/configs/conf.md) | `groot/vla/configs/conf.yaml` | 全局 Hydra 配置入口 |
| [vla](groot/vla/configs/model/dreamzero/vla.md) | `.../model/dreamzero/vla.yaml` | VLA 模型配置 |
| [wan_flow_matching_action_tf](groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf.md) | `.../action_head/wan_flow_matching_action_tf.yaml` | 动作头配置 |
| [identity](groot/vla/configs/model/dreamzero/backbone/identity.md) | `.../backbone/identity.yaml` | Identity 骨干配置 |
| [base transform](groot/vla/configs/model/dreamzero/transform/base.md) | `.../transform/base.yaml` | 基础变换配置 |
| [dreamzero_cotrain](groot/vla/configs/model/dreamzero/transform/dreamzero_cotrain.md) | `.../transform/dreamzero_cotrain.yaml` | 联合训练变换配置 |
| [droid_relative](groot/vla/configs/data/dreamzero/droid_relative.md) | `.../data/dreamzero/droid_relative.yaml` | DROID 数据配置 |
| [agibot_relative](groot/vla/configs/data/dreamzero/agibot_relative.md) | `.../data/dreamzero/agibot_relative.yaml` | AgiBot 数据配置 |
| [yam_relative](groot/vla/configs/data/dreamzero/yam_relative.md) | `.../data/dreamzero/yam_relative.yaml` | YAM 数据配置 |
| [base_48_wan_fine_aug_relative](groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.md) | `.../data/dreamzero/base_48_wan_fine_aug_relative.yaml` | 基础数据增强配置 |

### 数据处理 (groot/vla/data/)

| 文档 | 对应源文件 | 说明 |
|------|-----------|------|
| [lerobot](groot/vla/data/dataset/lerobot.md) | `.../dataset/lerobot.py` | LeRobot 数据集加载器 |
| [lerobot_sharded](groot/vla/data/dataset/lerobot_sharded.md) | `.../dataset/lerobot_sharded.py` | 分片 LeRobot 数据集 |
| [macro](groot/vla/data/dataset/macro.md) | `.../dataset/macro.py` | 数据集宏定义 |
| [metadata](groot/vla/data/dataset/metadata.md) | `.../dataset/metadata.py` | 数据集元数据管理 |
| [registry](groot/vla/data/dataset/registry.md) | `.../dataset/registry.py` | 数据集注册表 |
| [embodiment_tags](groot/vla/data/schema/embodiment_tags.md) | `.../schema/embodiment_tags.py` | 机器人本体标签定义 |
| [lerobot schema](groot/vla/data/schema/lerobot.md) | `.../schema/lerobot.py` | LeRobot 数据模式 |
| [transform/base](groot/vla/data/transform/base.md) | `.../transform/base.py` | 变换基类 |
| [transform/concat](groot/vla/data/transform/concat.md) | `.../transform/concat.py` | 多变换拼接 |
| [transform/language](groot/vla/data/transform/language.md) | `.../transform/language.py` | 语言变换 |
| [transform/state_action](groot/vla/data/transform/state_action.md) | `.../transform/state_action.py` | 状态与动作变换 |
| [transform/video](groot/vla/data/transform/video.md) | `.../transform/video.py` | 视频帧变换 |
| [conversion/gr1/constants](groot/vla/data/conversion/gr1/constants.md) | `.../conversion/gr1/constants.py` | GR1 数据转换常量 |
| [conversion/gr1/get_initial_actions](groot/vla/data/conversion/gr1/get_initial_actions.md) | `.../conversion/gr1/get_initial_actions.py` | GR1 初始动作提取 |

### 训练实验 (groot/vla/experiment/)

| 文档 | 对应源文件 | 说明 |
|------|-----------|------|
| [base](groot/vla/experiment/base.md) | `.../experiment/base.py` | 实验基类 |
| [experiment](groot/vla/experiment/experiment.md) | `.../experiment/experiment.py` | 训练实验主逻辑 |
| [utils](groot/vla/experiment/utils.md) | `.../experiment/utils.py` | 实验工具函数 |

### 核心模型 - DreamZero (groot/vla/model/dreamzero/)

| 文档 | 对应源文件 | 说明 |
|------|-----------|------|
| [base_vla](groot/vla/model/dreamzero/base_vla.md) | `.../dreamzero/base_vla.py` | VLA 基类 (Backbone + ActionHead) |
| [wan_flow_matching_action_tf](groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.md) | `.../action_head/wan_flow_matching_action_tf.py` | Flow Matching 动作头 |
| [base_backbone](groot/vla/model/dreamzero/backbone/base_backbone.md) | `.../backbone/base_backbone.py` | 骨干网络基类 |
| [identity](groot/vla/model/dreamzero/backbone/identity.md) | `.../backbone/identity.py` | Identity 骨干 (直通) |
| [dreamzero_cotrain](groot/vla/model/dreamzero/transform/dreamzero_cotrain.md) | `.../transform/dreamzero_cotrain.py` | 联合训练数据变换 |
| [common transform](groot/vla/model/dreamzero/transform/common.md) | `.../transform/common.py` | 通用模型变换 |

### 核心模块 (groot/vla/model/dreamzero/modules/)

| 文档 | 对应源文件 | 说明 |
|------|-----------|------|
| [wan_video_dit](groot/vla/model/dreamzero/modules/wan_video_dit.md) | `.../modules/wan_video_dit.py` | DiT 主体架构 |
| [wan_video_dit_action_casual_chunk](groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.md) | `.../modules/wan_video_dit_action_casual_chunk.py` | 动作因果分块 DiT |
| [wan_video_image_encoder](groot/vla/model/dreamzero/modules/wan_video_image_encoder.md) | `.../modules/wan_video_image_encoder.py` | CLIP 图像编码器 |
| [wan_video_text_encoder](groot/vla/model/dreamzero/modules/wan_video_text_encoder.md) | `.../modules/wan_video_text_encoder.py` | UMT5 文本编码器 |
| [wan_video_vae](groot/vla/model/dreamzero/modules/wan_video_vae.md) | `.../modules/wan_video_vae.py` | 视频 VAE 编解码器 |
| [wan_video_camera_controller](groot/vla/model/dreamzero/modules/wan_video_camera_controller.md) | `.../modules/wan_video_camera_controller.py` | 摄像头控制器 |
| [wan2_1_attention](groot/vla/model/dreamzero/modules/wan2_1_attention.md) | `.../modules/wan2_1_attention.py` | Wan2.1 注意力实现 |
| [wan2_1_submodule](groot/vla/model/dreamzero/modules/wan2_1_submodule.md) | `.../modules/wan2_1_submodule.py` | Wan2.1 子模块 |
| [attention](groot/vla/model/dreamzero/modules/attention.md) | `.../modules/attention.py` | 通用注意力模块 |
| [cudnn_attention](groot/vla/model/dreamzero/modules/cudnn_attention.md) | `.../modules/cudnn_attention.py` | cuDNN 加速注意力 |
| [flow_match_scheduler](groot/vla/model/dreamzero/modules/flow_match_scheduler.md) | `.../modules/flow_match_scheduler.py` | Flow Matching 调度器 |
| [flow_unipc_multistep_scheduler](groot/vla/model/dreamzero/modules/flow_unipc_multistep_scheduler.md) | `.../modules/flow_unipc_multistep_scheduler.py` | UniPC 多步调度器 |
| [vram_management](groot/vla/model/dreamzero/modules/vram_management.md) | `.../modules/vram_management.py` | 显存管理 |
| [utils](groot/vla/model/dreamzero/modules/utils.md) | `.../modules/utils.py` | 模块工具函数 |

### N1.5 策略模型 (groot/vla/model/n1_5/)

| 文档 | 对应源文件 | 说明 |
|------|-----------|------|
| [sim_policy](groot/vla/model/n1_5/sim_policy.md) | `.../n1_5/sim_policy.py` | 仿真策略封装 |
| [base_action_head](groot/vla/model/n1_5/action_head/base_action_head.md) | `.../action_head/base_action_head.py` | 动作头基类 |
| [action_encoder](groot/vla/model/n1_5/modules/action_encoder.md) | `.../modules/action_encoder.py` | 动作编码器 |

### 通用工具 (groot/vla/common/utils/)

| 文档 | 对应源文件 | 说明 |
|------|-----------|------|
| [config_utils](groot/vla/common/utils/io/config_utils.md) | `.../io/config_utils.py` | 配置工具 |
| [file_utils](groot/vla/common/utils/io/file_utils.md) | `.../io/file_utils.py` | 文件操作工具 |
| [hdf5_utils](groot/vla/common/utils/io/hdf5_utils.md) | `.../io/hdf5_utils.py` | HDF5 读写工具 |
| [json_utils](groot/vla/common/utils/io/json_utils.md) | `.../io/json_utils.py` | JSON 工具 |
| [print_utils](groot/vla/common/utils/io/print_utils.md) | `.../io/print_utils.py` | 输出格式化工具 |
| [termcolor](groot/vla/common/utils/io/termcolor.md) | `.../io/termcolor.py` | 终端颜色工具 |
| [array_tensor_utils](groot/vla/common/utils/misc/array_tensor_utils.md) | `.../misc/array_tensor_utils.py` | 数组/张量工具 |
| [functional_utils](groot/vla/common/utils/misc/functional_utils.md) | `.../misc/functional_utils.py` | 函数式工具 |
| [image_utils](groot/vla/common/utils/misc/image_utils.md) | `.../misc/image_utils.py` | 图像处理工具 |
| [misc_utils](groot/vla/common/utils/misc/misc_utils.md) | `.../misc/misc_utils.py` | 杂项工具 |
| [torch_utils](groot/vla/common/utils/misc/torch_utils.md) | `.../misc/torch_utils.py` | PyTorch 工具 |
| [video_utils](groot/vla/common/utils/misc/video_utils.md) | `.../misc/video_utils.py` | 视频处理工具 |
| [shape_utils](groot/vla/common/utils/data_structure/shape_utils.md) | `.../data_structure/shape_utils.py` | 形状操作工具 |
| [tree_utils](groot/vla/common/utils/data_structure/tree_utils.md) | `.../data_structure/tree_utils.py` | 树结构工具 |

### VLA 工具 (groot/vla/utils/)

| 文档 | 对应源文件 | 说明 |
|------|-----------|------|
| [action_args_override_utils](groot/vla/utils/action_args_override_utils.md) | `.../utils/action_args_override_utils.py` | 动作参数覆盖 |
| [timer](groot/vla/utils/timer.md) | `.../utils/timer.py` | 计时器 |

---

> **提示**: 索引中的链接指向 `tutorial/` 目录下对应的 Markdown 文件。这些文件将逐步创建和完善，具体进度请参考 [TODO.md](TODO.md)。
