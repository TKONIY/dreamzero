# 骨干网络（Backbone）配置索引

## 概述

`backbone/` 目录包含 DreamZero 骨干网络模块的配置文件。在 DreamZero 架构中，视觉编码由 WAN 扩散模型内部完成，因此骨干网络使用恒等（Identity）占位。

## 文件索引

- [identity.yaml](identity.md) -- Identity 骨干网络（无操作占位）

## 设计说明

DreamZero 不使用独立的视觉骨干网络（如 ViT、SigLIP 等），这与传统 VLA 模型（如 OpenVLA、RT-2）的设计不同。原因是 DreamZero 的视觉理解能力完全由 WAN 视频扩散模型的预训练权重提供，包括内置的图像编码器和 VAE。

如果需要添加独立的视觉编码器，可以在此目录下创建新的配置文件，并在 `vla.yaml` 中切换 backbone 选择。
