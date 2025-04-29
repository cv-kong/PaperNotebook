# 参考链接

- 链接: https://blog.csdn.net/Z960515/article/details/138952993

# 研究问题

- 由于GT融合数据的稀缺性，对融合模型的有效训练具有挑战性。



# 解决方法
- 提出了一种名为EMMA的新型自监督学习范式，旨在解决图像融合中真值缺失的问题
- 通过伪感知模块和感知损失分量有效地模拟感知成像过程，改进了传统融合损失中对融合图像和源输入之间域差异的不恰当处理。
- 提出的U - Fuser融合模块熟练地建模跨多个尺度的长、短程依赖关系来整合源信息。
- 方法在红外-可见光图像融合和医学图像融合中表现出优异的性能，这也被证明有利于下游的多模态目标检测和语义分割任务。
# 个人想法

- UNet进行图像融合，可替换Unet的模块，参考mmsegementation中的模型；替换理由为替换模型设计的优势以及为什么设计。https://github.com/open-mmlab/mmsegmentation

- 图像变换，常用的图像变换粗暴的添加进去，做消融实验验证加了比不加好
- 参考作者逻辑加一个图像增强模块，先训练好，再freeze。替换它的Ai,Av;选顶会的模型，不用训练直接用；省去训练Ai,Av；参考
- Ai,Vi本质是图像增强，目前top的图像增强网络是HVI。先训练后增强；

**HVI: A New color space for Low-light Image Enhancement**

- Paper: https://arxiv.org/abs/2502.20272
- Code: https://github.com/Fediory/HVI-CIDNet
- Demo: https://huggingface.co/spaces/Fediory/HVI-CIDNet_Low-light-Image-Enhancement_

- <font color="green">借鉴论文，应用到红外-可见光图像融合</font>

