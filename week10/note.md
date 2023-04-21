# 卷积神经网络

强烈推荐：[An Intuitive Explanation of Convolutional Neural Networks &#8211; Ujjwal Karn](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)

注：本笔记中的“卷积核”、“过滤器”、“算子”、“filter”在宏观上可以视作同一概念

## 能够实现卷积运算的函数：

python:conv_forward(需要实现)

TensorFlow：tf.nn.conv2D()

keras:Conv2D

## 垂直边缘检测  水平边缘检测

想一想什么情况能产生垂直边缘：左亮又暗和左暗右亮

同理，什么情况能产生水平边缘：上亮下暗和上暗下亮

![65FBC449BF6C2B84E917948485D7208D.png](D:\软件\tim\file\MobileFile\65FBC449BF6C2B84E917948485D7208D.png)

![3B7366F72A3682FE388346D269BB81AA.png](D:\软件\tim\file\MobileFile\3B7366F72A3682FE388346D269BB81AA.png)

如果我们不需要考虑正边缘和负边缘，那直接对特征矩阵去abs即可

但这确实能反映出当前边缘是正边缘还是负边缘

水平边缘检测当然也是同理，如果是一个3x3的卷积核，那么顶层为1，中间层为0，底层为-1

## 卷积核的给定

卷积核一般有两种获得方式：

1、手动给定

2、算法学习得到

### 手动给定

并不唯一，比如在边缘检测中，除了上面用到的过滤器，还有诸如下列过滤器

![](C:\Users\mings\AppData\Roaming\marktext\images\2023-04-17-21-49-07-image.png)

sobel过滤器：给中间行赋予了更大的权重，从而使得它更加稳定（？）

## 算法自主学习

通俗的来说，就是通过神经网络反向传播，找到最佳的卷积核

![](C:\Users\mings\AppData\Roaming\marktext\images\2023-04-17-21-59-51-image.png)

可以看到，不同的卷积核能够检测出不同角度的边缘，通过改变卷积核中元素的布局以及元  素的值，理论上可以实现360内任意偏向边缘的精准检测

参考：[边缘检测 - Scharr 滤波器_LqSilence的博客-CSDN博客](https://blog.csdn.net/LqSilence/article/details/104622926)

## 会产生的一些问题

1、每一次卷积都会使得图像变小，且变小的速度非常快

2、角落或者边界上的像素被使用的次数少很多，丢失了很多图片边界上的信息

可以使用**补白（padding）** 解决

补白其实就是在原始图像外围补一圈不会干扰检测结果的像素，

## 补白padding

- It allows you to use a CONV layer without necessarily shrinking the height and width of the volumes. This is important for building deeper networks, since otherwise the height/width would shrink as you go to deeper layers. An important special case is the "same" convolution, in which the height/width is exactly preserved after one layer.

- It helps us keep more of the information at the border of an image. Without padding, very few values at the next layer would be affected by pixels at the edges of an image.

![IMG_20230418_152131_edit_497853854233928.jpg](D:\软件\tim\file\MobileFile\IMG_20230418_152131_edit_497853854233928.jpg)

means that if the size of the filter is odd, we can use this method to solve this problem.

and the size of filter is usually odd

## 步幅stride

常规的卷积中，每次移动filter一个步长（即往右移一格）

![3B12E018522AA08DF1B31F7FE78731E9.png](D:\软件\tim\file\MobileFile\3B12E018522AA08DF1B31F7FE78731E9.png)

## 为什么the size of filter通常是奇数

1、为了使用same convolution对原始图像进行padding

2、奇数尺寸的filter会有一个中心点，对于视觉来说，有一个特殊点是很好的

## 三维卷积

![E84B3FDDBF8A957AF7910F020A72FDED.png](D:\软件\tim\file\MobileFile\E84B3FDDBF8A957AF7910F020A72FDED.png)

## 为什么这里要加偏差？？？

![](C:\Users\mings\AppData\Roaming\marktext\images\2023-04-18-20-23-54-image.png)

## 这个部分没看过，后面学到了再来看看

week10-1.7 卷积网络的一层-2:41

![](C:\Users\mings\AppData\Roaming\marktext\images\2023-04-18-20-26-56-image.png)

## 如果你的一层神经网络中10个3x3x3的过滤器，那么会有多少个参数？

![146A9C12CF6A6F2DAC503900F55A9CF1.png](D:\软件\tim\file\MobileFile\146A9C12CF6A6F2DAC503900F55A9CF1.png)

所以说，参数的个数只由filter的尺寸和数量决定，和初始图像的尺寸无关，无论是6x6还是6000x6000，只要filter的尺寸和个数一定，那么参数的个数也一定。这优于传统神经网络，参数数量大幅下降，从而可以处理大尺寸的图像

这个其实就是参数共享（parameter sharing）

## 激活层是什么？跳了一部分内容，后面补上

![](C:\Users\mings\AppData\Roaming\marktext\images\2023-04-18-21-00-45-image.png)

## 基本逻辑

一般从激活层得出下一层的步骤如下：

1、卷积

2、加偏差值

3、激活函数（RELU）

![8C760D969880A0FE9459199AF5D01C8C.png](D:\软件\tim\file\MobileFile\8C760D969880A0FE9459199AF5D01C8C.png)

红框圈出的部分是超参数，下面贴一个wiki的解释，简单来说，超参数就是在训练模型的过程中需要我们给定的参数，区别于例如节点权重这些通过训练得出的参数。

像这里的filter的尺寸和，都属于超参数。那么这些参数如何选定呢？后面来补充

## 超参数（后面会讲）

![](C:\Users\mings\AppData\Roaming\marktext\images\2023-04-18-22-17-42-image.png)

## 池化 pooling

### max pooling最大池化

常见的池化就是maxpooling，即规定一个尺寸的池化区域，去扫某一个激活状态，取当前区域内的最大值作为输出构成一个输出矩阵。

为什么我们能这么做？其实仔细想一想，这忽略了大部分信息。

首先想一想什么情况下卷积后的值会大。我们知道filter其实就是我们想要找到的特征，那么filter和激活状态的某一部分卷积后的值越大，说明该部分和filter的相似度越高，也就意味着这一块区域是特征的概率就越高。反之，则说明这块区域不太可能是特征。

按照吴恩达老师的说法，maxpooling的实际作用就是，如果在过滤器中提取到了某个特征，那么保留其最大值；如果没有提取到这个特征，可能在右上象限中不存在这个特征，那么其中的最大值也还是很小。

其实这段与其说是作用，不如说是选择maxpooling的原因

### average pooling 均值池化

通常来说，最大池化比均值池化用的更多，有一个例外是当神经网络深度非常大 时，用均值池化来合并表示

![E2F82D6A76DC2A4BDE0BD21A90E19584.png](D:\软件\tim\file\MobileFile\E2F82D6A76DC2A4BDE0BD21A90E19584.png)

上面卷积中提到的公式在池化中同样适用。因为卷积是相乘求和取出一个值，池化是限定区域取出一个值，所以输出个数是完全一样的。

如果输入是三维的，比如有两个channel，那么输出的channel也为2，这和卷积的过程有所区别，在卷积的过程中，输出的channel取决于filter的个数

### pooling中的超参数

pooling中的超参数有：

f：filter size

s：stride

pooling中通常不设置padding，但是有**例外（后面补充）**

一般来说，取f=2,d=2；或者f=3,s=2

## 相较于只用全连接层的神经网络，卷积神经网络有什么优点？

1、parameter sharing（参数共享）

![4{4Z{BPW`{U[]GXPUVM}2T2.png](D:\软件\tim\file\MobileFile\Image\4{4Z{BPW`{U[]GXPUVM}2T2.png)

也就是说，对于一个输入，可以使用一个filter对整个输入进行分析，不需要对不同的区域设置不同的filter

2、sparsity of connections（局部连接）

**再理解一下**

![](C:\Users\mings\AppData\Roaming\marktext\images\2023-04-19-16-57-12-image.png)

## 问题

### 几个layer的超参数怎么选定

吴恩达教授的意思是，这些超参已经有研究，可以套上去试一试。这个说法有些“结果指向性”了，还是得看看代码怎么写的

### 全连结层怎么工作

全连结层其实就是传统的nn，但是我跳了一部分，卷积的意义就是减少大量参数，并且仍然保留特征所需的参数

### 为什么几个layer层能找到特征

感觉和选定的filter有很大的关系

### 怎么做反向传播

不像nn公式那么清晰，感觉这个似有似无，全是卷积运算

- Explain the convolution operation
- Apply two different types of pooling operation
- Identify the components used in a convolutional neural network (padding, stride, filter, ...) and their purpose
- Build a convolutional neural network

## onehot编码（独热编码）

在很多机器学习任务重，特征并不总是连续值，也有可能是离散值，将这些数据用数字来表示，执行的效率会高很多。但是若是直接转换成数字的话，也不能直接用在分类器中，因为分类器往往默认数据是连续的、有序的。但是，直接数字并不是有序的，而是随机分配的。

为了解决上述问题，其中一种可能得解决方法是采用独热编码。

参考：[特征提取方法: one-hot 和 TF-IDF - ML小菜鸟 - 博客园 (cnblogs.com)](https://www.cnblogs.com/lianyingteng/p/7755545.html)
