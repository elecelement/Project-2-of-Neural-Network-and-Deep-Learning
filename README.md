# CIFAR-10 Classification and Batch Normalization Analysis

## 项目简介 / Project Overview

本项目包含两个部分：

1. **任务一：在 CIFAR-10 数据集上构建并优化神经网络模型**  
   使用卷积神经网络（CNN）对 CIFAR-10 图像进行分类，实验中尝试了不同网络结构、激活函数、损失函数和优化策略，并进行了模型训练与评估。

2. **任务二：Batch Normalization（BN）在网络训练中的效果分析**  
   基于 VGG-A 架构，分析添加 BN 层前后的网络训练效果，通过可视化 loss landscape 和梯度变化探究其优化机制。

## 项目结构 / Project Structure

```
data/                         # CIFAR-10 数据文件
network_for_CIFAR10/         # 完成任务一的代码
│   get_data.py              # 数据加载与预处理
│   model_odd.py             # 自定义 CNN 模型
│   train.py                 # 模型训练脚本（准确率92.4%）
│   train_with_another.py   # 另一个训练模型（准确率较低）
│   visualization.py        # 分类结果和特征可视化
│   cifar_net.pth           # 训练好的模型参数
│   存档                     # 其余模型的py程序 
reports/
│   figures/                 # 所有可视化图像文件（loss、accuracy 等）
VGG_BatchNorm/               # 完成任务二的代码
│   data/, models/, utils/  # VGG 模型构建与支持模块
│   VGG_Loss_Landscape.py   # loss landscape 可视化
│   visualize.py            # 梯度与损失可视化
│   visualize_landscape.py  # BN 前后 loss landscape 比较
│   *.txt                   # 各类 loss 与梯度记录文件
```

## 使用方法 / How to Run

1. 安装依赖（建议 Python 3.8+）：

```bash
pip install torch torchvision matplotlib numpy
```

2. 运行任务一训练脚本：

```bash
cd network_for_CIFAR10
python train.py
# 或使用 VGG 架构训练
python train_with_VGG.py
```

3. 运行任务二的分析脚本：

```bash
cd VGG_BatchNorm
python VGG_Loss_Landscape.py
python visualize.py
python visualize_landscape.py
```

## 结果可视化 / Results & Visualization

- 所有训练与对比结果（loss 曲线、loss landscape、梯度变化）已保存至 `reports/figures/`。
- 可视化示例包括：
  - CIFAR-10 分类准确率和损失曲线
  - VGG vs VGG+BN 的 loss landscape 比较图
  - 不同模型梯度变化对比图

## 模型与数据链接 / Model & Dataset Links

- 数据集：使用 PyTorch 自动下载的 CIFAR-10 数据（无需手动下载）
- 模型参数（已训练）：`network_for_CIFAR10/存档/cifar_net.pth`  
