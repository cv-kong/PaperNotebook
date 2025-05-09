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
- 作者用15个顶会融合后的数据作为GT，训练Ai，Av；因为图像输入输出有相同的size，所有选用unet；<font color="red">如果有不同的size呢？</font>
# 个人想法

- 先用一个预训练的模型来求解特征，预训练模型用的unet，该模型提取特征效果不是最佳，并且推理速度慢；
- unet需要图像输入输出有相同的size，训练出来的特征，并不能完美的模拟出真实传感器分布。替换成最新的backbone效果更好；<font color="red">打印出改进前和改进后的特征图</font>

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



# 摘要

- <font color="red">在别人基础下改进，摘要怎么描述???</font>
红外可见光图像，缺乏GT图像，主流且有效使用自监督方式进行融合。但该方法在特征提取和弱光照条件下，效果不佳,并且RGB和灰度图之间的转换会存在问题？？？。针对以上问题，我们在已有EMMA框架下改进，引入轻量级高效backbone，替换传统的Unet，解决输入尺度不同问题，并且强化特征提取；改进了伪感知模块，在I图像和V图像分策略优化，V图像进行增强特征提取，I图像简单特征提取；融合模块设计了混合注意力机制，加强特征提取一种基于自监督学习的新型融合方法；实验表明，该方法在红外可见光图像融合任务上取得了良好的效果。

# 介绍
- 介绍红外可见光图像融合的重要性，以及目前存在的问题
- 介绍EMMA框架，以及其存在的问题
  - V图像是RGB图像，存在光照问题，需要增强

# 动态非线性亮度映射（Dynamic Non-linear Lightness Mapping, DNLM）

## 1. 核心原理

### 1.1 动态参数化S形曲线
亮度变换公式：
$$ I_{out}(x,y) = \frac{1}{1 + e^{-k(x,y) \cdot (I_{in}(x,y) - \tau(x,y))}} $$

参数定义：
- $k(x,y)$：曲线陡峭度
  $$ k(x,y) = \alpha \cdot \frac{\sigma(x,y)}{\mu(x,y) + \epsilon} + \beta $$
- $\tau(x,y)$：动态阈值
  $$ \tau(x,y) = \gamma \cdot \mu(x,y) + \delta $$

### 1.2 语义引导约束
语义掩膜修正：
$$ k'(x,y) = k(x,y) \cdot (1 - \lambda \cdot S(x,y)) $$

## 2. 实现步骤

1. **局部统计计算**
   - 使用$15\times15$邻窗
   - 积分图加速

2. **曲线参数预测**
   - 轻量级CNN/LUT
   - 超网络动态生成

3. **像素级映射**
   - 逐像素S形变换
   - 导向滤波后处理

## 3. 优势对比

| 特性               | DNLM                          | 传统方法          |
|--------------------|-------------------------------|------------------|
| 自适应能力         | 像素级动态调整               | 全局固定参数      |
| 语义感知           | 支持区域抑制                | 无区分           |
| 计算效率           | 实时处理4K                  | 通常更快         |


  - EMMA中，存在伪感知模块backbone,Unet通过特征图可视化可知，细节特征提取的不佳；选用优化后的backbone，效果更好

# UNet作为Backbone的局限性及新型Backbone设计思路

## UNet的核心缺陷

### 1. 计算效率问题
- 对称编解码结构带来大量冗余计算
- 跳跃连接导致特征图尺寸频繁变化，内存访问效率低
- 浅层特征重复传递增加显存消耗

### 2. 特征融合瓶颈
- 简单的通道拼接(concat)操作限制多尺度特征交互
- 缺乏跨尺度注意力机制
- 深层语义信息与浅层细节融合不充分

### 3. 感受野局限
- 标准卷积核难以捕获长程依赖
- 下采样策略导致空间信息不可逆损失
- 对超大尺寸器官(如肝脏)分割效果下降

### 4. 动态适应性不足
- 固定网络结构难以适应不同模态数据
- 缺乏对病灶形状变化的鲁棒性处理

---

## 创新Backbone设计原则

### 1. 异构图灵架构
```python
class HeteroBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.branch1 = nn.Sequential(
            DepthwiseSepConv(in_c, out_c//2),
            ChannelAttention(out_c//2))
        self.branch2 = nn.Sequential(
            DilatedConv(in_c, out_c//4, dilation=3),
            SpatialAttention())
        self.branch3 = nn.Identity() if in_c==out_c else Conv1x1(in_c, out_c-out_c//2-out_c//4)
        
    def forward(self, x):
        return torch.cat([self.branch1(x), 
                        self.branch2(x),
                        self.branch3(x)], dim=1)
```
### 2. 多尺度感知模块
```python
class MSAP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scales = nn.ModuleList([
            nn.AvgPool2d(2**i, 2**i) for i in range(4)])
        self.fuse = nn.Conv2d(channels*4, channels, 1)
        
    def forward(self, x):
        feats = [F.interpolate(s(x), x.shape[2:]) for s in self.scales]
        return self.fuse(torch.cat(feats, dim=1))
```
  - EMMA中Uf模块，存在特征提取不足，无法提取到目标的完整信息


- 介绍本文的创新点，以及改进后的EMMA框架
- 介绍实验结果，以及与其他方法的对比

# 相关工作

- 介绍EMMA框架，以及其存在的问题
- 介绍其他红外可见光图像融合方法，以及存在的问题