# Classification

## Probabilistic Generative Model

### 分类与回归的不同

首先对于分类任务，很容易将其与回归任务联系起来。以二分类为例，对于某值空间的数据，将其划分为两个区域，指定为不同的类别，就算完成了我们的分类，以数学形式表达，就是寻找一个超平面$w^Tx+b=0$使得划分成立。然而，该方法在面对极端数据会出现不恰当的分类边界：

<img src="/Users/LightningX/Learning/ML2020/4.Classification/Note/截屏2020-08-30 14.00.35.png" alt="截屏2020-08-30 14.00.35" style="zoom:14%;" />

如图，当某些数据过于明显可分，决策边界为了减少Loss，会向着这些数据偏移，但明眼人都知道这种边界毫无意义。所以实际上使用回归任务的方式做分类就会涉及到损失函数定义不同带来的影响（回归衡量的是每个数据点到决策界的差值，数据距离越远，差值越大）。所以需要考虑设计某种函数，超过一个阈值后就给出确定的分类，或用类似的方法，以压缩Loss的影响。

### 生成模型与贝叶斯思想

<img src="/Users/LightningX/Learning/ML2020/4.Classification/Note/截屏2020-08-30 14.06.59.png" alt="截屏2020-08-30 14.06.59" style="zoom:14%;" />

考虑如上图的两个盒子。我们知道选取每个盒子（类）的概率与在每个盒子里头抽取出不同颜色球的概率。前者被称为先验概率Prior，后者被称为似然概率Likelihood。在此基础上，通过贝叶斯定理，我们在面对一个新的球时候，就可以利用这些概率，计算抽出的这个球应该来自于哪个箱子，所求的内容称为后验概率Posterior。

在生成模型中，我们计算的不仅仅是某些数据属于哪一类，而是通过计算该类的分布与数据条件分布去推算结果，因此，若我们能够得出这些概率，就完全可以得到这些数据的分布，从而进行数据生成。

### Case study: Pokemon分类预测

<img src="/Users/LightningX/Learning/ML2020/4.Classification/Note/截屏2020-08-30 14.27.18.png" alt="截屏2020-08-30 14.27.18" style="zoom:14%;" />

还是Pokemon的预测问题。只是这次预测的目标是给定某只的防御和特防向量，给出一个分类概率，进而确定分类。在本设定中，假设数据是服从高斯分布的，那么接下来的任务就是确定这群数据的一个代表高斯分布，从而计算出新数据的出现概率。

#### 似然函数求模型参数

因对于数据的采样独立，因此可以将需要计算的所有样本概率进行连乘组合，得到的L就是一个似然函数。易知当该似然函数最大的时候，这个高斯分布的均值与方差就是我们所求的分布参数。以对于水系的样本预测为例，79个样本的均值与方差计算如下（通过求导方法可以得到这个最大值，此处省略，可见机器学习白板推导中关于高斯判别分析的推导介绍）：

<img src="/Users/LightningX/Learning/ML2020/4.Classification/Note/截屏2020-08-30 14.37.02.png" alt="截屏2020-08-30 14.37.02" style="zoom:14%;" />

同样的方法可以对普通系神奇宝贝做一个计算，最终对于两个不同的类，得到两组不同的$\mu$与$\Sigma$。利用刚才提到的后验概率公式，对于一个输入向量$x$，能够计算出其属于不同类的概率（二分类里头以0.5作为分类界限），从而做出分类。决策边界如下：

<img src="/Users/LightningX/Learning/ML2020/4.Classification/Note/截屏2020-08-30 14.41.39.png" alt="截屏2020-08-30 14.41.39" style="zoom:14%;" />

不难看到，使用这两个特征的情况下效果并不好。为了提高效果，使用所有7个特征，准确率也不过在一半上下，这是没有意义的。

#### 简化模型

李老师提到一个观点，即高维情况下协方差矩阵的参数数量将急剧增加，因此这很有可能带来过拟合。基于此，实际使用中一般假设这些不同类的数据都符合一个相同方差的高斯分布，仅仅是均值不同。共享分布后，似然函数表示为所有不同类样本的概率连乘，从而得到协方差矩阵。两个分布的均值向量求法不变。

通过这一简单改进，模型的效果居然发生翻天覆地的变化：

<img src="/Users/LightningX/Learning/ML2020/4.Classification/Note/截屏2020-08-30 14.49.12.png" alt="截屏2020-08-30 14.49.12" style="zoom:14%;" />

使用两个特征时候，模型的决策边界形状发生了变化。使用全部七个特征的模型准确率大大提高了。

### 后验概率与线性模型的联系

对上一个实例中的概率，可作如下改写：
$$
\begin{align*}
P(C_1|x)&=\frac{P(x|C_1)P(C_1)}{P(x|C_1)P(C_1)+P(x|C_2)P(C_2)}
\\
&=\frac{1}{1+\frac{P(x|C_2)P(C_2)}{P(x|C_1)P(C_1)}}
\end{align*}
$$
记$z=ln\frac{P(x|C_1)P(C_1)}{P(x|C_2)P(C_2)}$，则上式改写为：
$$
P(C_1|x)=\frac{1}{1+exp(-z)}=\sigma(z)
$$
这里引入了Sigmoid函数$\sigma(\cdot)$。

继续推导：
$$
\begin{align*}
z&=ln\frac{P(x|C_1)}{P(x|C_2)}+ln\frac{P(C_1)}{P(C_2)}
\\
&=ln\frac{\frac{1}{(2\pi)^{p/2}|\Sigma_1|^{1/2}}exp(-\frac{1}{2}(x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1))}{\frac{1}{(2\pi)^{p/2}|\Sigma_2|^{1/2}}exp(-\frac{1}{2}(x-\mu_2)^T\Sigma_2^{-1}(x-\mu_2))}+ln\frac{N_1}{N_2}
\\
&=ln\frac{|\Sigma_2|^{1/2}}{|\Sigma_1|^{1/2}}exp(-\frac{1}{2}(x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1))+\frac{1}{2}(x-\mu_2)^T\Sigma_2^{-1}(x-\mu_2)))+ln\frac{N_1}{N_2}
\\
&=ln\frac{|\Sigma_2|^{1/2}}{|\Sigma_1|^{1/2}}-\frac{1}{2}(x^T\Sigma_1^{-1}x-2\mu_1^T\Sigma_1^{-1}x+\mu_1^T\Sigma_1^{-1}\mu_1-x^T\Sigma_2^{-1}x+2\mu_2^T\Sigma_2^{-1}x-\mu_2^T\Sigma_2^{-1}\mu_2)+ln\frac{N_1}{N_2}
\end{align*}
$$
根据我们的简化假设，当不同类间数据符合一个相同协方差的高斯分布时，$\Sigma_1=\Sigma_2$，则有：
$$
\begin{align*}
z&=-\frac{1}{2}(-2\mu_1^T\Sigma^{-1}x+\mu_1^T\Sigma^{-1}\mu_1+2\mu_2^T\Sigma^{-1}x-\mu_2^T\Sigma^{-1}\mu_2)+ln\frac{N_1}{N_2}
\\
&=(\mu_1^T-\mu_2^T)\Sigma^{-1}x-\frac{1}{2}(\mu_1^T\Sigma^{-1}\mu_1-\mu_2^T\Sigma^{-1}\mu_2)+ln\frac{N_1}{N_2}
\end{align*}
$$
实际上令$w^T=(\mu_1^T-\mu_2^T)\Sigma^{-1}$，$b=-\frac{1}{2}(\mu_1^T\Sigma^{-1}\mu_1-\mu_2^T\Sigma^{-1}\mu_2)+ln\frac{N_1}{N_2}$，上述式子即表达为$w^Tx+b$，则$P(C_1|x)=\sigma(w^Tx+b)$。

想象线性模型的分类边界是一个超平面$w^Tx+b=0$，在Sigmoid函数的设定中，当$w^Tx_i+b>0$时$P(C_1|x_i)>0.5$归为一类，反之为另一类。而通过以上推导，我们发现简化高斯分布情况下，二分类生成模型和Sigmoid函数形式相同，这也就是为什么我们简化模型的决策边界是一条“直线”（超平面）了。

## Logistic Regression

续前，从概率生成模型中，我们引入了Sigmoid函数，接下来将正式将其表述。

***Step1: Function Set***

需要找到一个概率$P_{w,b}(C_1|x)$，当其不小于0.5时输出类别1，否则为类别二。因此可以使用Sigmoid函数$\sigma(z)=\frac{1}{1+exp(-z)},z=w\cdot x+b$。实际上最终得到的是$f_{w,b}(x)=\sigma(\sum_i w_ix_i+b)$。需要注意的是，分类问题规定输出类别为0或1，则输出值在0-1之间。

***Step2: Goodness of a Function***

假设所有$(x^i,C)$都是由给定$x_i$的$C$概率分布$f_{w,b}(x)=P_{w,b}(C_1|x)$得到，那么损失函数可定义为所有类别$C_1$的概率连乘。例如对于数据$(x^1,C_1),(x^2,C_1),(x^3,C_2),...,(x^N,C_1)$，可得目标函数为：
$$
L(w,b)=f_{w,b}(x^1)f_{w,b}(x^2)(1-f_{w,b}(x^3))\cdots f_{w,b}(x^N)
$$
并且这里涉及的最优参数为使得L最大的$w$与$b$：$w^*,b^*=arg\max_{w,b}L(w,b)$。

考虑到连乘的优化困难，可以取对数转化为连加。

在实际使用中，因为二分类问题，可以对训练数据的标签加以限定，例如令类别1标签为1，类别2标签为0。通过该设定，可以将目标函数特化，从而带有良好数学性质。

首先，最大似然函数求参数可以通过取对数拆分乘积，并加上负号转化为最小化问题：
$$
w^*,b^*=arg\min_{w,b}-lnL(w,b)
$$
此时每一个$l nf_{w,b}(x^i)$在规定类别标签情况下，均可写为$y^ilnf(x^i)+(1-y^i)ln(1-f(x^i))$，因此总体目标函数表述为：
$$
\sum_n-[\hat{y}^nlnf_{w,b}(x^n)+(1-\hat{y}^n)ln(1-f_{w,b}(x^n))]
$$
实际上，若考虑两个伯努利分布如下：

<img src="/Users/LightningX/Learning/ML2020/4.Classification/Note/截屏2020-09-06 20.39.22.png" alt="截屏2020-09-06 20.39.22" style="zoom:50%;" />

我们的目标函数就呈现为这两个分布的交叉熵！

***Step3: Find the best function***

求导的过程不详述，实际上就是逻辑回归的求导。实际利用到的公式为：
$$
\begin{align*}
\frac{\partial ln\sigma(z)}{\partial w_i}&=1-\sigma(z)
\\
\frac{\partial ln(1-\sigma(z))}{\partial w_i}&=\sigma(z)(1-\sigma(z))
\end{align*}
$$
<img src="/Users/LightningX/Learning/ML2020/4.Classification/Note/截屏2020-09-06 20.43.44.png" alt="截屏2020-09-06 20.43.44" style="zoom:33%;" />

实际代入后发现，对于$w$的更新过程，当预测值$f$与真实值$\hat{y}$差别越大时，求导出来的结果也越“大”，使得更新更加明显，这符合我们对于梯度下降的一般直觉。

### 为什么不用平方损失函数？

<img src="/Users/LightningX/Learning/ML2020/4.Classification/Note/截屏2020-09-06 20.50.49.png" alt="截屏2020-09-06 20.50.49" style="zoom:50%;" />

通过求导发现，逻辑回归和线性回归得到的梯度更新公式完全一致，那么逻辑回归看作取值特殊的线性回归，为什么不能用平方损失函数呢？

实际上这和逻辑回归的问题背景有极大相关。我们来看使用平方损失函数的结果：

<img src="/Users/LightningX/Learning/ML2020/4.Classification/Note/截屏2020-09-06 20.53.03.png" alt="截屏2020-09-06 20.53.03" style="zoom:33%;" />

因为分类问题的真实值固定为0或1。当取值为1时有预测值靠近0或1两种情况（sigmoid函数会让取值压缩到这两个值附近），此时他们的梯度都趋于0，从而无法做到有效的更新。对于取值0也有同样的结论。

<img src="/Users/LightningX/Library/Application Support/typora-user-images/截屏2020-09-06 20.55.10.png" alt="截屏2020-09-06 20.55.10" style="zoom:33%;" />

以一张图来表示，当使用平方损失函数的时候，在广大区域的梯度其实是很平的，无法做到快速的收敛。

### 生成 or 判别

对于判别模型（如LR回归），其通过不断优化确定模型参数（$w$与$b$）。而生成模型则需要找到所有数据的均值与方差向量/矩阵，并归纳形式求出模型参数。同样的模型与数据，会产生不一样的函数参数结果。在课程ppt举的一个例子中，判别式模型反而有可能得到更高的准确率。个人猜测，这是由于生成式模型对于数据的假设过强导致的。当然生成式模型有个巨大优点：在带有概率分布假设的情况下，通过极少量数据就可以得到模型（判别式模型是数据驱动），且对噪声更加鲁棒。