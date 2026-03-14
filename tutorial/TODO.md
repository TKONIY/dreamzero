# DreamZero 代码教程 - 进度跟踪
> 总计: ~107 个源文件需要文档化
> 创建日期: 2026-03-14
> 完成日期: 2026-03-14

## 进度概览
| 模块 | 文件数 | 已完成 | 状态 |
|------|--------|--------|------|
| root | 3 | 3 | ✅ 完成 |
| eval_utils | 3 | 3 | ✅ 完成 |
| scripts/train | 5 | 5 | ✅ 完成 |
| scripts/data | 3 | 3 | ✅ 完成 |
| scripts (misc) | 2 | 2 | ✅ 完成 |
| groot/vla/common/utils/io | 6 | 6 | ✅ 完成 |
| groot/vla/common/utils/misc | 6 | 6 | ✅ 完成 |
| groot/vla/common/utils/data_structure | 2 | 2 | ✅ 完成 |
| groot/vla/configs | 13 | 13 | ✅ 完成 |
| groot/vla/data/dataset | 5 | 5 | ✅ 完成 |
| groot/vla/data/schema | 2 | 2 | ✅ 完成 |
| groot/vla/data/transform | 5 | 5 | ✅ 完成 |
| groot/vla/data/conversion | 2 | 2 | ✅ 完成 |
| groot/vla/experiment | 3 | 3 | ✅ 完成 |
| groot/vla/model/n1_5 | 3 | 3 | ✅ 完成 |
| groot/vla/model/dreamzero | 20 | 20 | ✅ 完成 |
| groot/vla/utils | 2 | 2 | ✅ 完成 |

## 基础文档
- [x] `tutorial/00_background_knowledge.md` — 背景知识
- [x] `tutorial/00_reading_guide.md` — 阅读指南
- [x] `tutorial/01_inference_walkthrough.md` — 推理流程自顶向下详解

## 文件清单（按模块）

### root/
- [x] `socket_test_optimized_AR.py` → `tutorial/root/socket_test_optimized_AR.md`
- [x] `test_client_AR.py` → `tutorial/root/test_client_AR.md`
- [x] `pyproject.toml` → `tutorial/root/pyproject.md`

### eval_utils/
- [x] `eval_utils/policy_client.py` → `tutorial/eval_utils/policy_client.md`
- [x] `eval_utils/policy_server.py` → `tutorial/eval_utils/policy_server.md`
- [x] `eval_utils/run_sim_eval.py` → `tutorial/eval_utils/run_sim_eval.md`

### scripts/
- [x] `scripts/train/droid_training.sh` → `tutorial/scripts/train/droid_training.md`
- [x] `scripts/train/droid_training_lora.sh` → `tutorial/scripts/train/droid_training_lora.md`
- [x] `scripts/train/droid_training_full_finetune.sh` → `tutorial/scripts/train/droid_training_full_finetune.md`
- [x] `scripts/train/agibot_training.sh` → `tutorial/scripts/train/agibot_training.md`
- [x] `scripts/train/yam_training.sh` → `tutorial/scripts/train/yam_training.md`
- [x] `scripts/data/convert_droid.py` → `tutorial/scripts/data/convert_droid.md`
- [x] `scripts/data/convert_lerobot_to_gear.py` → `tutorial/scripts/data/convert_lerobot_to_gear.md`
- [x] `scripts/data/convert_agibot.py` → `tutorial/scripts/data/convert_agibot.md`
- [x] `scripts/compare_loss.py` → `tutorial/scripts/compare_loss.md`
- [x] `scripts/open_loop_yam.py` → `tutorial/scripts/open_loop_yam.md`

### groot/vla/common/
- [x] `groot/vla/common/utils/io/config_utils.py` → `tutorial/groot/vla/common/utils/io/config_utils.md`
- [x] `groot/vla/common/utils/io/file_utils.py` → `tutorial/groot/vla/common/utils/io/file_utils.md`
- [x] `groot/vla/common/utils/io/hdf5_utils.py` → `tutorial/groot/vla/common/utils/io/hdf5_utils.md`
- [x] `groot/vla/common/utils/io/json_utils.py` → `tutorial/groot/vla/common/utils/io/json_utils.md`
- [x] `groot/vla/common/utils/io/print_utils.py` → `tutorial/groot/vla/common/utils/io/print_utils.md`
- [x] `groot/vla/common/utils/io/termcolor.py` → `tutorial/groot/vla/common/utils/io/termcolor.md`
- [x] `groot/vla/common/utils/misc/array_tensor_utils.py` → `tutorial/groot/vla/common/utils/misc/array_tensor_utils.md`
- [x] `groot/vla/common/utils/misc/functional_utils.py` → `tutorial/groot/vla/common/utils/misc/functional_utils.md`
- [x] `groot/vla/common/utils/misc/image_utils.py` → `tutorial/groot/vla/common/utils/misc/image_utils.md`
- [x] `groot/vla/common/utils/misc/misc_utils.py` → `tutorial/groot/vla/common/utils/misc/misc_utils.md`
- [x] `groot/vla/common/utils/misc/torch_utils.py` → `tutorial/groot/vla/common/utils/misc/torch_utils.md`
- [x] `groot/vla/common/utils/misc/video_utils.py` → `tutorial/groot/vla/common/utils/misc/video_utils.md`
- [x] `groot/vla/common/utils/data_structure/shape_utils.py` → `tutorial/groot/vla/common/utils/data_structure/shape_utils.md`
- [x] `groot/vla/common/utils/data_structure/tree_utils.py` → `tutorial/groot/vla/common/utils/data_structure/tree_utils.md`

### groot/vla/configs/
- [x] `groot/vla/configs/conf.yaml` → `tutorial/groot/vla/configs/conf.md`
- [x] `groot/vla/configs/model/dreamzero/vla.yaml` → `tutorial/groot/vla/configs/model/dreamzero/vla.md`
- [x] `groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf.yaml` → `tutorial/groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf.md`
- [x] `groot/vla/configs/model/dreamzero/backbone/identity.yaml` → `tutorial/groot/vla/configs/model/dreamzero/backbone/identity.md`
- [x] `groot/vla/configs/model/dreamzero/transform/base.yaml` → `tutorial/groot/vla/configs/model/dreamzero/transform/base.md`
- [x] `groot/vla/configs/model/dreamzero/transform/dreamzero_cotrain.yaml` → `tutorial/groot/vla/configs/model/dreamzero/transform/dreamzero_cotrain.md`
- [x] `groot/vla/configs/data/dreamzero/droid_relative.yaml` → `tutorial/groot/vla/configs/data/dreamzero/droid_relative.md`
- [x] `groot/vla/configs/data/dreamzero/agibot_relative.yaml` → `tutorial/groot/vla/configs/data/dreamzero/agibot_relative.md`
- [x] `groot/vla/configs/data/dreamzero/yam_relative.yaml` → `tutorial/groot/vla/configs/data/dreamzero/yam_relative.md`
- [x] `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml` → `tutorial/groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.md`
- [x] `groot/vla/configs/deepspeed/zero2.json` → `tutorial/groot/vla/configs/deepspeed/zero2.md`
- [x] `groot/vla/configs/deepspeed/zero2_offload.json` → `tutorial/groot/vla/configs/deepspeed/zero2_offload.md`
- [x] `groot/vla/configs/deepspeed/zero3.json` → `tutorial/groot/vla/configs/deepspeed/zero3.md`

### groot/vla/data/
- [x] `groot/vla/data/dataset/lerobot.py` → `tutorial/groot/vla/data/dataset/lerobot.md`
- [x] `groot/vla/data/dataset/lerobot_sharded.py` → `tutorial/groot/vla/data/dataset/lerobot_sharded.md`
- [x] `groot/vla/data/dataset/macro.py` → `tutorial/groot/vla/data/dataset/macro.md`
- [x] `groot/vla/data/dataset/metadata.py` → `tutorial/groot/vla/data/dataset/metadata.md`
- [x] `groot/vla/data/dataset/registry.py` → `tutorial/groot/vla/data/dataset/registry.md`
- [x] `groot/vla/data/schema/embodiment_tags.py` → `tutorial/groot/vla/data/schema/embodiment_tags.md`
- [x] `groot/vla/data/schema/lerobot.py` → `tutorial/groot/vla/data/schema/lerobot.md`
- [x] `groot/vla/data/transform/base.py` → `tutorial/groot/vla/data/transform/base.md`
- [x] `groot/vla/data/transform/concat.py` → `tutorial/groot/vla/data/transform/concat.md`
- [x] `groot/vla/data/transform/language.py` → `tutorial/groot/vla/data/transform/language.md`
- [x] `groot/vla/data/transform/state_action.py` → `tutorial/groot/vla/data/transform/state_action.md`
- [x] `groot/vla/data/transform/video.py` → `tutorial/groot/vla/data/transform/video.md`
- [x] `groot/vla/data/conversion/gr1/constants.py` → `tutorial/groot/vla/data/conversion/gr1/constants.md`
- [x] `groot/vla/data/conversion/gr1/get_initial_actions.py` → `tutorial/groot/vla/data/conversion/gr1/get_initial_actions.md`

### groot/vla/experiment/
- [x] `groot/vla/experiment/base.py` → `tutorial/groot/vla/experiment/base.md`
- [x] `groot/vla/experiment/experiment.py` → `tutorial/groot/vla/experiment/experiment.md`
- [x] `groot/vla/experiment/utils.py` → `tutorial/groot/vla/experiment/utils.md`

### groot/vla/model/n1_5/
- [x] `groot/vla/model/n1_5/sim_policy.py` → `tutorial/groot/vla/model/n1_5/sim_policy.md`
- [x] `groot/vla/model/n1_5/action_head/base_action_head.py` → `tutorial/groot/vla/model/n1_5/action_head/base_action_head.md`
- [x] `groot/vla/model/n1_5/modules/action_encoder.py` → `tutorial/groot/vla/model/n1_5/modules/action_encoder.md`

### groot/vla/model/dreamzero/
- [x] `groot/vla/model/dreamzero/base_vla.py` → `tutorial/groot/vla/model/dreamzero/base_vla.md`
- [x] `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` → `tutorial/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.md`
- [x] `groot/vla/model/dreamzero/backbone/base_backbone.py` → `tutorial/groot/vla/model/dreamzero/backbone/base_backbone.md`
- [x] `groot/vla/model/dreamzero/backbone/identity.py` → `tutorial/groot/vla/model/dreamzero/backbone/identity.md`
- [x] `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py` → `tutorial/groot/vla/model/dreamzero/transform/dreamzero_cotrain.md`
- [x] `groot/vla/model/dreamzero/transform/common.py` → `tutorial/groot/vla/model/dreamzero/transform/common.md`
- [x] `groot/vla/model/dreamzero/modules/wan_video_dit.py` → `tutorial/groot/vla/model/dreamzero/modules/wan_video_dit.md`
- [x] `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` → `tutorial/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.md`
- [x] `groot/vla/model/dreamzero/modules/wan_video_image_encoder.py` → `tutorial/groot/vla/model/dreamzero/modules/wan_video_image_encoder.md`
- [x] `groot/vla/model/dreamzero/modules/wan_video_text_encoder.py` → `tutorial/groot/vla/model/dreamzero/modules/wan_video_text_encoder.md`
- [x] `groot/vla/model/dreamzero/modules/wan_video_vae.py` → `tutorial/groot/vla/model/dreamzero/modules/wan_video_vae.md`
- [x] `groot/vla/model/dreamzero/modules/wan_video_camera_controller.py` → `tutorial/groot/vla/model/dreamzero/modules/wan_video_camera_controller.md`
- [x] `groot/vla/model/dreamzero/modules/wan2_1_attention.py` → `tutorial/groot/vla/model/dreamzero/modules/wan2_1_attention.md`
- [x] `groot/vla/model/dreamzero/modules/wan2_1_submodule.py` → `tutorial/groot/vla/model/dreamzero/modules/wan2_1_submodule.md`
- [x] `groot/vla/model/dreamzero/modules/attention.py` → `tutorial/groot/vla/model/dreamzero/modules/attention.md`
- [x] `groot/vla/model/dreamzero/modules/cudnn_attention.py` → `tutorial/groot/vla/model/dreamzero/modules/cudnn_attention.md`
- [x] `groot/vla/model/dreamzero/modules/flow_match_scheduler.py` → `tutorial/groot/vla/model/dreamzero/modules/flow_match_scheduler.md`
- [x] `groot/vla/model/dreamzero/modules/flow_unipc_multistep_scheduler.py` → `tutorial/groot/vla/model/dreamzero/modules/flow_unipc_multistep_scheduler.md`
- [x] `groot/vla/model/dreamzero/modules/vram_management.py` → `tutorial/groot/vla/model/dreamzero/modules/vram_management.md`
- [x] `groot/vla/model/dreamzero/modules/utils.py` → `tutorial/groot/vla/model/dreamzero/modules/utils.md`

### groot/vla/utils/
- [x] `groot/vla/utils/action_args_override_utils.py` → `tutorial/groot/vla/utils/action_args_override_utils.md`
- [x] `groot/vla/utils/timer.py` → `tutorial/groot/vla/utils/timer.md`
