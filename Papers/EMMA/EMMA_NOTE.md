# 参考链接

- 链接: https://blog.csdn.net/Z960515/article/details/138952993

# 研究问题

- 由于GT融合数据的稀缺性，对融合模型的有效训练具有挑战性。
- 单个传感器，难以捕捉到目标的完整信息。不存在传感器能够同时捕捉到目标的所有特征
- 基于生成式的方法，认为融合图像属于相同分布，具有解释性差，难以训练等问题
- 常规的优化方法，求解的是原图像图像特征直接与融合图像特征的映射，忽略了源图像之间的相关性



# 解决方法

- 基于自监督学习的方法，提出了一种名为EMMA的新型自监督学习范式，旨在解决图像融合中真值缺失的问题
- 通过伪感知模块和感知损失分量有效地模拟感知成像过程，改进了传统融合损失中对融合图像和源输入之间域差异的不恰当处理。
- 提出的U - Fuser融合模块熟练地建模跨多个尺度的长、短程依赖关系来整合源信息。
- 方法在红外-可见光图像融合和医学图像融合中表现出优异的性能，这也被证明有利于下游的多模态目标检测和语义分割任务。
- 作者用15个顶会融合后的数据作为GT，训练Ai，Av；因为图像输入输出有相同的size，所有选用unet；<font color="red">如果有不同的size呢</font>
# 个人想法

- 先用一个预训练的模型来求解特征，预训练模型用的unet，该模型提取特征效果不是最佳，并且推理速度慢；
- unet训练出来的特征，并不能完美的模拟出真实传感器分布。替换成最新的backbone效果更好；<font color="red">打印出改进前和改进后的特征图</font>

- UNet进行图像融合，可替换Unet的模块，参考mmsegementation中的模型；替换理由为替换模型设计的优势以及为什么设计。https://github.com/open-mmlab/mmsegmentation

- 图像变换，常用的图像变换粗暴的添加进去，做消融实验验证加了比不加好
- 参考作者逻辑加一个图像增强模块，先训练好，再freeze。替换它的Ai,Av;选顶会的模型，不用训练直接用；省去训练Ai,Av；
- Ai,Vi本质是图像增强，目前top的图像增强网络是HVI。先训练后增强；训练得到传感器特性
- 原始模型，不是端到端的训练。

- 链接： https://docs.ultralytics.com/zh/models/yolo12/#overview
- YOLOV12 使用Area attention；更大感受野，更小计算量
- 常规的融合算法，都只强调算法效果，未强调性能。目前Unet和Ufuser结构，效率上差。https://zhuanlan.zhihu.com/p/24876772668
![效率](images\注意力机制效率问题.png)

- 论文使用多头注意力机制提取全局特征 https://zhuanlan.zhihu.com/p/647747759、https://zhuanlan.zhihu.com/p/509582087

**HVI: A New color space for Low-light Image Enhancement**

- Paper: https://arxiv.org/abs/2502.20272
- Code: https://github.com/Fediory/HVI-CIDNet
- Demo: https://huggingface.co/spaces/Fediory/HVI-CIDNet_Low-light-Image-Enhancement_

- <font color="green">借鉴论文，应用到红外-可见光图像融合</font>

**YOLOv12: Attention-Centric Real-Time Object Detectors**

- 参考： https://zhuanlan.zhihu.com/p/24876772668
- paper：https://arxiv.org/pdf/2502.12524

# 创新
- propose a simple yet efficient area attention module(A2), which maintains a large receptive field while reducing the computational complexity of attention in a very simple way, thus enhancing speed.