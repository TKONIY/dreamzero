# DreamZero 代码教程

## 项目简介

DreamZero（"World Action Models Are Zero-Shot Policies"）是由 NVIDIA Gear Lab 开发的视觉-语言-动作（VLA）模型框架。该项目的核心思想是将世界模型（World Model）直接用作零样本策略（Zero-Shot Policy），通过视频生成式的自回归推理来预测机器人动作。

## 教程模块索引

| 模块 | 路径 | 说明 |
|------|------|------|
| [根目录文件](root/index.md) | `root/` | 推理服务端、测试客户端、项目配置 |
| [评估工具](eval_utils/index.md) | `eval_utils/` | 策略客户端/服务端、仿真评估 |
| [训练与数据脚本](scripts/index.md) | `scripts/` | 训练启动脚本、数据转换脚本 |
| [核心模块](groot/index.md) | `groot/` | VLA 模型、数据管线、实验框架、配置系统 |

## 推荐起点

- **[推理流程详解](01_inference_walkthrough.md)** -- 从推理入口 `socket_test_optimized_AR.py` 出发，自顶向下逐层剖析完整推理调用链

## 基础文档

- [背景知识](00_background_knowledge.md) -- DreamZero 涉及的核心概念与技术背景
- [阅读指南](00_reading_guide.md) -- 推荐的教程阅读顺序与学习路径

## 进度跟踪

- [TODO.md](TODO.md) -- 文档化进度与文件清单
