## 小型实验记录 - Text Classification

#### 模型1

使用milti-filter CNN，同时设置三个卷积层，将input同时通过三个卷积层与三个池化层后，将三个output拼接得到fully connected layers的输入，最终输出一个四维向量。训练的速度非常慢，跑一个epoch需要接近两分钟。最终效果不是很好，在遍历3个epoch后验证集的准确率就收敛在64%左右，且验证集的loss在跑三个epoch后随遍历epoch数目而不断上升，这说明3个epoch后就会出现overfitting。在测试集上的表现：跑3个epoch后准确率为65.1%，跑10个epoch后准确率为64.4%。

