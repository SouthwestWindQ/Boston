## 小型实验记录 - Text Classification

#### 模型1

使用milti-filter CNN，同时设置三个卷积层，将input同时通过三个卷积层与三个池化层后，将三个output拼接得到fully connected layers的输入，最终输出一个四维向量。embedding没有训练。在cpu上训练的速度非常慢，跑一个epoch需要接近两分钟。最终效果不是很好，在遍历3个epoch后验证集的准确率就收敛在64%左右，且验证集的loss在跑三个epoch后随遍历epoch数目而不断上升，这说明3个epoch后就会出现overfitting。在测试集上的表现：跑3个epoch后准确率为65.1%，跑10个epoch后准确率为64.4%。

#### 模型2

与模型1的唯一区别就在于embedding放入神经网络中进行训练。经训练，验证集的准确率相比模型1有很大进步（大约提高10个百分点），最终测试集的准确率在遍历150个epoch左右后收敛于77%左右。模型2使用gpu进行训练，遍历一个epoch只需要1秒的时间。最终在测试集上的表现：

<table style="text-align:center;">
  <tr>
  	<th>epoch</th>
    <th>accuracy</th>
  </tr>
  <tr>
  	<td>10</td>
    <td>73.1%</td>
  </tr>
  <tr>
  	<td>50</td>
    <td>74.9%</td>
  </tr>
   <tr>
  	<td>100</td>
    <td>73.7%</td>
  </tr>
  <tr>
  	<td>300</td>
    <td>75.8%</td>
  </tr>
  <tr>
  	<td>500</td>
    <td>77.7%</td>
  </tr>
  <tr>
  	<td>1000</td>
    <td>78%</td>
  </tr>
</table>

观察epoch和验证集准确率的关系统计图（见"`./multi-filter CNN(trained embedding)/`"文件夹），最后发现epoch取200应该是比较合适的，因为epoch过大之后验证集准确率的提升并不显著。



