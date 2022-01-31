from turtle import forward
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as fun

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(13,14),
            nn.ReLU(),
            nn.Linear(14,1)
        )
    def forward(self,x):
        # x = fun.relu(self.net1(x))
        # x = fun.dropout(x, p=0.5)
        # x = fun.relu(self.net2(x))
        # x = fun.dropout(x, p=0.5)
        return self.net(x)
        
    
table_x,table_y = datasets.load_boston(return_X_y=True)
scaler = MinMaxScaler()
tensor_x = torch.from_numpy(scaler.fit_transform(table_x))
# tensor_y = torch.from_numpy(table_y[:,np.newaxis])
tensor_y = torch.from_numpy(scaler.fit_transform(table_y[:,np.newaxis]))
myTrainDataset = TensorDataset(tensor_x[0:350, : ], tensor_y[0:350, : ])
myTestDataset = TensorDataset(tensor_x[351:506, : ], tensor_y[351:506, : ])
tr_data = DataLoader(dataset=myTrainDataset, batch_size=22, shuffle=True)
tt_data = DataLoader(dataset=myTestDataset, shuffle=False)

model = MyModel().to('cpu')
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
model.net = model.net.double()
# model.net2 = model.net2.double()
# model.net3 = model.net3.double()

iter_loss = []
batch_loss = []
epoN = 30
for epoch in range(epoN):
    model.train()
    for x,y in tr_data:
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
print(iter_loss[epoN-1])
        
model.eval()
total_loss = 0 
for x,y in tt_data:
    x,y = x.to('cpu'), y.to('cpu')
    with torch.no_grad():  #禁止计算梯度，只计算output(速度加快)
        pred = model(x)
        # print(x)
        # print(scaler.inverse_transform(pred), scaler.inverse_transform(y))
        loss = criterion(pred, y)
        total_loss += loss.cpu().item()*len(x)
        
avg_loss = total_loss / len(tt_data.dataset)
print(avg_loss)

# #x1 = np.array(table['B'])
# x2 = np.array(table['LSTAT'])
# y = np.array(table['MEDV'])
# #plt.scatter(x1,y)
# plt.scatter(x2,y)
# plt.show()'''