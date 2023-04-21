## week01

### RELU：？？？

激活函数？是激活函数是RELU还是RELU是激活函数？

相对于sigmoid有什么提升？

为什么我们需要激活函数？没有激活函数会发生什么？

通俗的来说，激活函数是用来解决非线性问题拟合的，y=AF(Wx)本身Wx是一个线性方程，如果我们需要用他拟合一个非线性问题，我们就可以套一层AF，将这个y=Wx掰弯

一般在CNN中，我们使用relu;在RNN中，我们使用relu或者tanh

### 隐藏单元：？？？

hidden unit，我理解这里的隐藏其实就是

### supervised learning：监督学习

### 为什么神经网络这么流行

因为大数据使用成为可能，对于小规模数据来说，神经网络和传统学习方式的性能相差有限，甚至一个SVM的性能优于更大规模的神经网络，当然是需要一个优秀组件和算法的SVM。所以当数据集相当大时，传统学习方式就难以处理，此时就需要神经网络

### Algorithm:

eg. sigmoid->Relu

sigmoid？(0,1)

<img src="file:///C:/Users/mings/AppData/Roaming/marktext/images/2023-04-10-16-16-05-image.png" title="" alt="" width="175">

<img src="file:///C:/Users/mings/AppData/Roaming/marktext/images/2023-04-10-14-41-21-image.png" title="" alt="" width="358">

Relu

<img src="file:///C:/Users/mings/AppData/Roaming/marktext/images/2023-04-10-14-51-16-image.png" title="" alt="" width="298">

在箭头所指区域，斜率会接近0，使得学习变得非常缓慢，因为梯度下降时，梯度接近0，参数会变得非常缓慢，使得学习也变得很慢，将sigmoid函数转换为Relu函数能将“梯度下降法”运行地更快  

## week2

### 正向传播？反向传播？

![](C:\Users\mings\AppData\Roaming\marktext\images\2023-04-11-14-47-01-image.png)

看到P14为止，我的理解，这个知识一个正向传播和反向传播的例子，这里所有的计算过程都是为了用代码实现计算斜率这一功能。（后面可能还会添加一些别的理解）

还有一点，我们首先计算了dJ/dv，这一式子可以拿来辅助计算dJ/du，因为dJ/du根据导数的链式法则equals to (dJ/dv)*(dv/du)，所以只需要额外计算dv/du即可

### 不用for循环遍历

用**向量化技术**(vetorization)来代替for loop，能够加速计算。

从编译的角度来说，其实就是把跳转的时间给避免了，计算的时间是相同的，而跳转的时间是远大于计算的时间的

![](C:\Users\mings\AppData\Roaming\marktext\images\2023-04-12-15-48-36-image.png)

### logistic回归——线性回归

显然logistic回归和线性回归不同，那么主要的不同点在哪里？logistic回归是在线性回归的基础上加了一个sigmoid函数,那么为什么我们能够套一个sigmoid函数呢？

### parameter b and w

![B674854982C320E6F522731231AA1E52.png](D:\软件\tim\file\MobileFile\B674854982C320E6F522731231AA1E52.png)

#### loss function 损失函数:衡量了在单个训练样本上的表现

就是描述y_pre和y有多相近

一般不用<img src="file:///C:/Users/mings/AppData/Roaming/marktext/images/2023-04-10-16-33-06-image.png" title="" alt="" width="183">因为优化问题会变成非凸(non-convex)，导致有多个局部最优解。

后面会有详细介绍

事实上是使用这个损失函数<img src="file:///C:/Users/mings/AppData/Roaming/marktext/images/2023-04-10-16-37-06-image.png" title="" alt="" width="293">，为什么呢（或者说这个损失函数能够产生什么样的效果）？

可以分析一下这个损失函数

![](C:\Users\mings\AppData\Roaming\marktext\images\2023-04-10-16-56-24-image.png)

后面会详细讲到为什么用这个函数而不用其他函数（事实上能够产生这种效果的损失函数非常多） 

但是吧，最大似然估计再看一下，忘了说是

![IMG_2122(20230414-131400).PNG](D:\软件\tim\file\MobileFile\IMG_2122(20230414-131400).PNG)



## cost function 基于参数的总成本

![IMG_2108(20230411-111843).PNG](D:\软件\tim\file\MobileFile\IMG_2108(20230411-111843).PNG)

我们想要的就是**cost尽可能小的参数W和b**

## 怎么去找到最优的W和b？

可以使用梯度下降算法，逼近最优解

## 梯度下降算法

![9CD45900156B089103ED9C367B6BBDE2.png](D:\软件\tim\file\MobileFile\9CD45900156B089103ED9C367B6BBDE2.png)

![3122B6D90EB9E8D5CBE5B286D9E8377C.png](D:\软件\tim\file\MobileFile\3122B6D90EB9E8D5CBE5B286D9E8377C.png)

单个训练样本的logistic回归的梯度下降法

![CC50E87479044C2A66BF61C24F61886D.png](D:\软件\tim\file\MobileFile\CC50E87479044C2A66BF61C24F61886D.png)

对于训练集

![660ACB0B03C4334BDCA3F81DCC6526A9.png](D:\软件\tim\file\MobileFile\660ACB0B03C4334BDCA3F81DCC6526A9.png)

## 用np中的向量值函数代替loop

np.exp  np.log np.abs np.

![](C:\Users\mings\AppData\Roaming\marktext\images\2023-04-12-16-03-49-image.png)

原理是什么？

向量化技术[1]Array programming with NumPy: *https://www.nature.com/articles/s41586-020-2649-2*

## python中的broadcast

general Principle

![74D00ABC4397839F0DB64060243FCF20.png](D:\软件\tim\file\MobileFile\74D00ABC4397839F0DB64060243FCF20.png)

### 一种情况

b是一个实数

z=np.dot(x,y)+b

此时，python会自动把b转换成一个1*n的向量

![E3E61444DA60EE237D548FF0DE792260.png](D:\软件\tim\file\MobileFile\E3E61444DA60EE237D548FF0DE792260.png)

### 另一种情况

![](C:\Users\mings\AppData\Roaming\marktext\images\2023-04-13-15-31-05-K[D1Z0MAQ77V1YM(102{[TS.png)

A是一个3x4的矩阵，cal是一个1x4的矩阵，理论上是没有矩阵除法的，但是np可以直接这样计算，这就是broadcast

还有一点就是cal后面建议跟一个reshape(1,4)。首先因为当数据量大的时候，我们并不一定能确定得到的矩阵的大小，所以reshape能够确保矩阵是我们需要的大小

## 用numpy的向量化技术来化简for-loop的整理

![F36FDE3721FB032D07FB2A4B61EFAC71.png](D:\软件\tim\file\MobileFile\F36FDE3721FB032D07FB2A4B61EFAC71.png)

## 避免使用"rank 1 array"

![](C:\Users\mings\AppData\Roaming\marktext\images\2023-04-13-16-48-39-image.png)

it is different with both col-vector and raw-vector, a and a.T is the same one

![](C:\Users\mings\AppData\Roaming\marktext\images\2023-04-13-16-50-06-image.png)

so we need to avoid using this data structure instead of using vector

![](C:\Users\mings\AppData\Roaming\marktext\images\2023-04-13-16-54-15-image.png)

and np.random.randn(1,5) is just the same, including other function that can form a normal vector
