**1.(2%) 請比較實作的 generative model 及 logistic regression 的準確率，何者較佳？請解釋為何有這種情況？**

比较发现，生成式模型的准确率：

```Python
Training accuracy: 0.8699130050132704
```

而LR的准确率为：

```Python
Training accuracy: 0.8836166291214418
Development accuracy: 0.8733873940287504
```

生成式模型准确率略低于LR回归，个人认为这是由于生成式模型对数据的强假设造成的。比如当数据不完全是某个假设分布的时候，完全依靠数据本身计算出的LR比起模拟数据分布的假设，按理说会有更符合数据集的结果。LR回归的求解过程本就是在目前的数据基础上求损失函数最小的一组参数，因此在训练集的准确率必然是最优解，而生成模型仅仅是一种假设，基于当前数据推算出背后可能的隐藏分布，所以并不存在某种“最小化损失”的目的。

**2.(2%) 請實作 logistic regression 的正規化 (regularization)，並討論其對於你的模型準確率的影響。接著嘗試對正規項使用不同的權重 (lambda)，並討論其影響。(有關 regularization 請參考 https://goo.gl/SSWGhf p.35)**

在LR回归中，我直接于提供的范例代码的cross_entropy函数中加入L2惩罚项，然而loss出现剧烈震荡，且最终提交结果并未得到更好的值。实际上我认为，LR回归是一种线性回归，所以其一定存在全局最优，因此使用正规化反而可能令某些参数无法被设置为正确的值，从而需要更多的迭代轮次，因此在相同次数下效果更差了。

**3.(1%) 請說明你實作的 best model，其訓練方式和準確率為何？**

best model中我仅增加了训练轮次从10至100，并略微修改了批次大小，从8变为16，且learning rate从0.2改为0.1。最终准确率为：

```Python
Training accuracy: 0.8859307802580381
Development accuracy: 0.8763361592333211
```



**4.(1%) 請實作輸入特徵標準化 (feature normalization)，並比較是否應用此技巧，會對於你的模型有何影響。**

仅以普通LR回归做比较，当未做标准化时，得到结果如下：

```Python
Training accuracy: 0.7654925250870367
Development accuracy: 0.7640987836343531
```

训练过程如图：

<img src="/Users/LightningX/Learning/ML2020/4.Classification/Code/loss.png" alt="loss" style="zoom:72%;" />

<img src="/Users/LightningX/Learning/ML2020/4.Classification/Code/acc.png" alt="acc" style="zoom:72%;" />

容易看到，未加入标准化的训练过程十分曲折，因为不同维度的数据大小差异较大，从而导致对loss的影响不同，收敛过程曲折。而加入标准化后结果为：

```Python
Training accuracy: 0.8836166291214418
Development accuracy: 0.8733873940287504
```

<img src="/Users/LightningX/Learning/ML2020/4.Classification/Code/loss_normalized.png" alt="loss_normalized" style="zoom:72%;" />

<img src="/Users/LightningX/Learning/ML2020/4.Classification/Code/acc_normalized.png" alt="acc_normalized" style="zoom:72%;" />

通过标准化，搜索过程被极大地平滑，并且经过很少的轮次就收敛到了较好的结果。因此最好结果必须运用标准化方法。

