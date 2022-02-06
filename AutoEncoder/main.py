import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.conv1 = nn.Conv2d(1,16,3)
        self.conv2 = nn.Conv2d(16,16,3)
        self.fc1 = nn.Linear(400,128)
        self.fc2 = nn.Linear(128,32)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1,self.flatten_size(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def flatten_size(self,x):
        sizes = x.size()[1:]
        features = 1
        for s in sizes:
            features *= s
        return features

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(32,128),
            nn.ReLU(),
            nn.Linear(128,784)
        )
    def forward(self,x):
        return self.net(x)

train_loader = DataLoader(
    datasets.MNIST('data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1037,),(0.3081,))
        ])),
    batch_size=64, shuffle=True, drop_last=False
)
test_loader = DataLoader(
    datasets.MNIST('data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1037,),(0.3081,))
        ])),
    batch_size=64, shuffle=True, drop_last=False
)
        
epoN = 30
encoder = Encoder().to('cpu')
decoder = Decoder().to('cpu')
criterion = nn.MSELoss()
optimizer1 = torch.optim.Adam(encoder.parameters())
optimizer2 = torch.optim.Adam(decoder.parameters())
iter_loss, test_loss = [], []
for epoch in range(epoN):
    encoder.train()
    decoder.train()
    batch_loss = []
    progress_bar = tqdm(train_loader, total=len(train_loader))
    for x,y in progress_bar:
        x = x.to('cpu')
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        code = encoder(x)
        pred = decoder(code)
        loss = criterion(pred,x.view(-1,784))
        batch_loss.append(loss.data.numpy())
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        progress_bar.set_description('Train | epoch[{}/{}] | loss: {:.3f}'.format(epoch+1,epoN,np.average(np.array(batch_loss))))
    iter_loss.append(np.average(np.array(batch_loss)))
    
encoder.eval()
decoder.eval()
batch_loss = []
progress_bar = tqdm(test_loader, total=len(test_loader))
for x,y in progress_bar:
    x = x.to('cpu')
    with torch.no_grad():
        code = encoder(x)
        pred = decoder(code)
        loss = criterion(pred,x.view(-1,784))
        batch_loss.append(loss.data.numpy())
        progress_bar.set_description('Test  | loss: {:.3f}'.format(np.average(np.array(batch_loss))))
test_loss.append(np.average(np.array(batch_loss)))

mnist = datasets.MNIST('data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1037,),(0.3081,))
        ]))
fig,ax = plt.subplots(nrows=2,ncols=10,sharex='all',sharey='all')
ax = ax.flatten()
for i in range(10):
    img = mnist[i][0].view(28,28)
    ax[i].set_title('Pic {}'.format(i+1))
    ax[i].imshow(img,cmap='gray')
for i in range(10):
    with torch.no_grad():
        code = encoder(mnist[i][0].view(1,1,28,28))
        img = decoder(code).view(28,28)
        ax[10+i].set_title("Pic {}'".format(i+1))
        ax[10+i].imshow(img,cmap='gray')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
        
        
        
