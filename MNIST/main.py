from enum import EnumMeta
import torch, sys
import numpy as np
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as fun
from sklearn.datasets import fetch_california_housing
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from torchvision import datasets, transforms

class MyModel(nn.Module):
    def __init__(self, reluN):
        super(MyModel,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784,reluN),
            nn.ReLU(),
            nn.Linear(reluN,10),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        # x = fun.relu(self.net1(x))
        # x = fun.dropout(x, p=0.5)
        # x = fun.relu(self.net2(x))
        # x = fun.dropout(x, p=0.5)
        return self.net(x.view(-1,784))
        
tr_data = DataLoader(
    datasets.MNIST('data', train=True, download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1037,),(0.3081,))
        ])),
    batch_size=64, shuffle=True,
)
tt_data = DataLoader(
    datasets.MNIST('data', train=False, download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1037,),(0.3081,))
        ])),
    batch_size=64, shuffle=True,
)
# table_x,table_y = fetch_california_housing(return_X_y=True)
# scaler = MinMaxScaler()
# tensor_x = torch.from_numpy(scaler.fit_transform(table_x))
# tensor_y = torch.from_numpy(table_y[:,np.newaxis])
# # tensor_y = torch.from_numpy(scaler.fit_transform(table_y[:,np.newaxis]))
# myTrainDataset = TensorDataset(tensor_x[0:14450, : ], tensor_y[0:14450, : ])
# myTestDataset = TensorDataset(tensor_x[14451:20640, : ], tensor_y[14451:20640, : ])
# tr_data = DataLoader(dataset=myTrainDataset, batch_size=22, shuffle=True)
# tt_data = DataLoader(dataset=myTestDataset, shuffle=False)

iter_loss = []
batch_loss = []
avg_list = []
acc_list = []
epoN = 20
model = MyModel(256).to('cpu')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
# model.net2 = model.net2.double()
# model.net3 = model.net3.double()
sum = 0
for epoch in range(epoN):
    model.train()
    print(epoch+1)
    for x,y in tr_data:
        sum += 64
        print(sum)
        optimizer.zero_grad()
        x,y = x.to('cpu'),y.to('cpu')
        pred = model(x)
        loss = criterion(pred, y)
        batch_loss.append(loss.data.numpy())
        loss.backward()
        optimizer.step()
    iter_loss.append(np.average(np.array(batch_loss)))

    # x = np.arange(epoN)
    # y = np.array(iter_loss)
    # plt.scatter(x,y)
    # plt.show()
    # print(iter_loss[epoN-1])
        
    model.eval()
    total_loss = 0 
    cor,tot = 0,0
    for x,y in tt_data:
        x,y = x.to('cpu'), y.to('cpu')
        with torch.no_grad():  #禁止计算梯度，只计算output(速度加快)
            pred = model(x)
            for i in range(min(64,pred.shape[0])):
                max1,max2 = 0,pred[i][0]
                for j in range(1,10):
                    if pred[i][j]>max2:
                        max1 = j
                        max2 = pred[i][j]
                print(max1,y[i])
                tot += 1
                if max1==y[i]:
                    cor += 1
            # print(x)
            # print(scaler.inverse_transform(pred), scaler.inverse_transform(y))
            loss = criterion(pred, y)
            total_loss += loss.cpu().item()*len(x)
            
    avg_loss = total_loss / len(tt_data.dataset)
    avg_list.append(avg_loss)
    print(tot,cor)
    acc_list.append(cor/tot)
    print(avg_loss)
    del model
    model = MyModel(256).to('cpu')

x = np.array([i for i in range(1,21)])
# y1 = np.array(iter_loss)
y2 = np.array(acc_list)
# plt.scatter(x,y1)
plt.scatter(x,y2)
plt.show()

x = np.array([i for i in range(1,21)])
y1 = np.array(iter_loss)
y2 = np.array(avg_list)
plt.scatter(x,y1)
plt.scatter(x,y2)
plt.show()
        
# z = np.array(avg_list)
# x_list, y_list = [], []
# for i in range(1,31):
#     for j in range(250):
#         x_list.append(i)
#         y_list.append(j)
# x,y = np.array(x_list),np.array(y_list)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(x, y, z, cmap=cm.jet, linewidth=0.01)
# plt.show()

# #x1 = np.array(table['B'])
# x2 = np.array(table['LSTAT'])
# y = np.array(table['MEDV'])
# #plt.scatter(x1,y)
# plt.scatter(x2,y)
# plt.show()'''

# delta = 0.1
# x = np.array([1,2,3,4])
# y = np.array([1,2,3,4])
# # X, Y = np.meshgrid(x, y)
# z = np.array([])
# # x=X.flatten()
# # y=Y.flatten()
# # z=Z.flatten()
 
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(x, y, z, cmap=cm.jet, linewidth=0.01)
# plt.show()
# plt.contour(X,Y,Z)
# plt.show()