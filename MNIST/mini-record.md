## 小型实验记录 - MNIST

这里记录一些我采用不同模型尝试MNIST的实验结果。

首先采用了单层的神经网络进行机器学习，激活函数使用ReLU，latent dimension设为256。测试集的准确率在跑了10个epoch左右便趋于收敛，大致收敛在98%左右，是一个比较好的结果。测试集的loss随遍历epoch数目的上升呈增大趋势，这证明遍历过多的epoch将导致严重的overfitting。

然后我稍微调整了神经网络的结构，神经网络的层数增加为3层，hidden layer均采用ReLU作为激活函数，latent dimension均为256，结构采用最普通的fully connected feedforward network。事实证明层数的增加并不能带来准确率的显著提升，在遍历15个epoch后，测试集的准确率也大致收敛于98%左右。这说明，在深度学习中，fully connected的结构往往不能带来更好的效果，需要采用一些更好的结构，如CNN。

几天后我尝试用CNN做实验，结构是convolutional layer + pooling + convolutional layer + pooling + 3个fully connected layer，激活函数均选用ReLU，kernel size设为 $3\times3$，stride采用默认值1。实验结果表明，在遍历5个epoch左右后，测试集的准确率大致收敛于99%，且测试集的loss随epoch数目变多是趋于稳定而不是逐渐上升，这说明在手写数字识别这个任务上，CNN相比fully connected network拥有更好的效果。将kernel size从 $3\times3$ 调整为 $5\times5$ 后实验结果大致一致。

