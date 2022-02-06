## 小型实验记录 - AutoEncoder

#### 模型

Encoder采用CNN结构：卷积层 $\to$ 池化层 $\to$ 卷积层 $\to$ 池化层 $\to$ 一层fully connected layer。激活函数均选用ReLU，filters数目均为16，kernel size设为 $3\times3$ ，output的向量维数为32维，其它参数详见程序。Decoder采用简单的fully connected network，hidden dimension设为128，激活函数为ReLU。

#### 超参选取

在跑了50个epoch后发现：训练集和测试集的loss均随遍历epoch数目的增加而下降，但是测试集的loss在遍历30个epoch后下降速率明显变慢，所以最终决定遍历epoch数目为30。

#### 实验结果

选用decoder的output和encoder的input的所有像素点的MSE作为损失函数。最终测试集的loss（MSE）为0.115左右，encoder的input和decoder的output的对比图一并存放在AutoEncoder文件夹中（遍历1个epoch和30个epoch）。实验发现，遍历1个epoch后decoder的output已经能大致看清轮廓，而遍历30个epoch后decoder的output相比前者只是看起来更清晰了一些而已。经询问可能是学习率较大或batch_size较小所致。
