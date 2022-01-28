from turtle import forward
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from sklearn import datasets
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(13,32),
            nn.ReLU(),
            nn.Linear(32,1),
        )
    def forward(self,x):
        return self.net(x)
    
table_x,table_y = datasets.load_boston(return_X_y=True)
tensor_x = torch.from_numpy(table_x)
tensor_y = torch.from_numpy(table_y).unsqueeze(1)
myDataset = TensorDataset(tensor_x, tensor_y)
tr_data = DataLoader(dataset=myDataset, batch_size=22, shuffle=True)

model = MyModel().to('cpu')
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
model.net = model.net.double()

for epoch in range(100):
    model.train()
    for x,y in tr_data:
        optimizer.zero_grad()
        x,y = x.to('cpu'),y.to('cpu')
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
model.eval()
total_loss = 0 
tt_data = DataLoader(dataset=myDataset, shuffle=False)
for x,y in tt_data:
    x,y = x.to('cpu'), y.to('cpu')
    with torch.no_grad():  #禁止计算梯度，只计算output(速度加快)
        pred = model(x)
        # print(x)
        print(pred)
        # print(y)
        loss = criterion(pred, y)
        total_loss += loss.cpu().item()*len(x)
        
avg_loss = total_loss / len(tt_data.dataset)
print(avg_loss)
cnt = 0
for paras in model.parameters():
    cnt += 1
    if cnt==4:
        print(paras)

# #x1 = np.array(table['B'])
# x2 = np.array(table['LSTAT'])
# y = np.array(table['MEDV'])
# #plt.scatter(x1,y)
# plt.scatter(x2,y)
# plt.show()