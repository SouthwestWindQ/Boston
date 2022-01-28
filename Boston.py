from turtle import forward
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(13,14),
            nn.ReLU(),
            nn.Linear(14,1)
        )
    def forward(self,x):
        return self.net(x)
    
table_x,table_y = datasets.load_boston(return_X_y=True)
scaler = MinMaxScaler()
tensor_x = torch.from_numpy(scaler.fit_transform(table_x))
tensor_y = torch.from_numpy(scaler.fit_transform(table_y[:,np.newaxis]))
myTrainDataset = TensorDataset(tensor_x[0:350, : ], tensor_y[0:350, : ])
myTestDataset = TensorDataset(tensor_x[351:506, : ], tensor_y[351:506, : ])
tr_data = DataLoader(dataset=myTrainDataset, batch_size=22, shuffle=True)
tt_data = DataLoader(dataset=myTestDataset, shuffle=False)

model = MyModel().to('cpu')
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)
model.net = model.net.double()

iter_loss = []
batch_loss = []
for epoch in range(500):
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

x = np.arange(500)
y = np.array(iter_loss)
plt.scatter(x,y)
plt.show()
        
model.eval()
total_loss = 0 
for x,y in tt_data:
    x,y = x.to('cpu'), y.to('cpu')
    with torch.no_grad():  #禁止计算梯度，只计算output(速度加快)
        pred = model(x)
        # print(x)
        print(pred, y)
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