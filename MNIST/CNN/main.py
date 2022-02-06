from enum import EnumMeta
from re import M
import torch, sys
import numpy as np
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from torchvision import datasets, transforms
from tqdm import tqdm   #python进度条！

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN,self).__init__()
        self.conv1 = nn.Conv2d(1,16,5)
        self.conv2 = nn.Conv2d(16,16,5)
        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
train_loader = DataLoader(
    datasets.MNIST('data', train=True, download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1037,),(0.3081,))
        ])),
    batch_size=64, shuffle=True, drop_last=False
) #drop_last指的是，如果batch_size不整除数据总量，那余下的部分是否丢弃
test_loader = DataLoader(
    datasets.MNIST('data', train=False, download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1037,),(0.3081,))
        ])),
    batch_size=64, shuffle=True, drop_last=False
)

iter_loss = []
batch_loss = []
avg_list = []
acc_list = []
epoN = 20
model = MyCNN().to('cpu')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
# model.net2 = model.net2.double()
# model.net3 = model.net3.double()
sum = 0
for epoch in range(epoN):
    model.train()
    progress_bar = tqdm(train_loader, total =len(train_loader)) #进度条，使进度的显示更清晰，学习！
    cor,tot = 0,0
    for x,y in progress_bar:
        optimizer.zero_grad()
        x,y = x.to('cpu'),y.to('cpu')
        pred = model(x)
        loss = criterion(pred, y)
        batch_loss.append(loss.data.numpy())
        loss.backward()
        optimizer.step()

        cor += (pred.max(1)[1] == y).sum().item()  #学习这种简省的写法！
        tot += x.shape[0]
        progress_bar.set_description('Train | epoch [{}/{}] | loss: {:.3f} | acc: {:.3f}'.format(epoch+1, epoN, loss.item(), cor/tot))

    iter_loss.append(np.average(np.array(batch_loss)))
        
    model.eval()
    total_loss = 0 
    cor,tot = 0,0
    progress_bar = tqdm(test_loader, total =len(test_loader))
    for x,y in progress_bar:
        x,y = x.to('cpu'), y.to('cpu')
        with torch.no_grad():  #禁止计算梯度，只计算output(速度加快)
            pred = model(x)
            # for i in range(min(64,pred.shape[0])):
            #     max1,max2 = 0,pred[i][0]
            #     for j in range(1,10):
            #         if pred[i][j]>max2:
            #             max1 = j
            #             max2 = pred[i][j]
            #     print(max1,y[i])
            #     tot += 1
            #     if max1==y[i]:
            #         cor += 1

            loss = criterion(pred, y)
            total_loss += loss.cpu().item()*len(x)
            cor += (pred.max(1)[1] == y).sum().item()
            tot += x.shape[0]
            progress_bar.set_description('Test  | epoch [{}/{}] | loss: {:.3f} | acc: {:.3f}'.format(epoch+1, epoN, loss.item(), cor/tot))
            
    avg_loss = total_loss / len(test_loader.dataset)
    avg_list.append(avg_loss)
    acc_list.append(cor/tot)

x = np.array([i for i in range(1,epoN+1)])
# y1 = np.array(iter_loss)
y2 = np.array(acc_list)
# plt.scatter(x,y1)
plt.scatter(x,y2)
plt.show()

x = np.array([i for i in range(1,epoN+1)])
y1 = np.array(iter_loss)
y2 = np.array(avg_list)
plt.scatter(x,y1)
plt.scatter(x,y2)
plt.show()