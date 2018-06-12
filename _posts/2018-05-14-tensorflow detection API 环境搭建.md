---
layout: post
title: "tensorflow detection API 环境搭建"
date: 2018-5-14
description: "tensorflow detection API 环境搭建"
tag: tensorflow 
--- 

本文主要包括tensorflow detection API 环境搭建

### tensorflow detection API 简介

tensorflow object detection API是一个开源的基于tensorflow的框架，使得创建，训练以及应用目标检测模型变得简单。
我们已经发现这个代码对计算机视觉研究需要很有用，我们主要利用这个API在医学图像中检测病灶。

### 开发环境搭建（现有tensorflow=1.2.0，cuda=8.0,cudnn=5.1,anconda(python=3.6)

## 1. tensorflow更新
tensorflow detection API 需要tensorflow=1.4.0，于是我们就更新了一下tensorflow

        $ pip install --upgrade tensorflow

比较难受的事情出现了，cudnn=5.1版本不支持tensorflow=1.4,于是又更新了cudnn, 还有python=3.6不能安装tensorflow=1.4,于是利用conda建立了python=3.5的环境。
这里cudnn更新和python=3.5环境搭建就不在这里具体描述了，之前的博客都有详细描述。

## 2. 下载tensorflow detection API

<a target="_blank" href="https://github.com/tensorflow/models/"> https://github.com/tensorflow/models </a>
从github上下载项目（右上角“Clone or download”-"DownloadZIP"），下载到本地目录（避免中文），解压。

## 3. Protobuf 安装与配置

 Tensorflow Object Detection API 用 Protobufs 来配置模型和训练参数. 在用这个框架之前,必须先编译Protobuf 库，
 切换到这个目录下： tensorflow/models/research/.

        $ cd ~/tensorflow/models/research/
        $ protoc object_detection/protos/*.proto --python_out=.
        
 添加环境变量
        
        $ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

注意: 这条命令在新打开的终端中需要重新执行一次才会在新终端中生效，如果不想那么麻烦，就用下面的命令编辑 ~/.bashrc 文件，把上面的语句添加到末尾.

        $ gedit ~/.bashrc

## 4. 测试

        % cd ~/tensorflow/models/research/object_detection

打开jupyter notebook

运行object_detection_tutorial.ipynb

如果没有错，并成功检测出dog,说明环境搭建成功！至此Tensorflow object detection API 的环境搭建与测试工作完成。

下一步我们可以在此基础上对代码进行适当的修改，可以用已有的模型来检测自己的图片，甚至视频，并输出结果。

在进一步，可以用自己标注的数据集进行训练与评估。

