import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, subsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        nn.init.kaiming_normal_(
            self.conv1.weight, mode='fan_out', nonlinearity='relu'
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        nn.init.kaiming_normal_(
            self.conv2.weight, mode='fan_out', nonlinearity='relu'
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.subsample = subsample
    
    def forward(self, x):
        id = x if self.subsample is None else self.subsample(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += id
        x = F.relu(x)
        return x


class MyResNet(nn.Module):
    def __init__(self):
        super(MyResNet, self).__init__()
        self.N = 18
        self.in_channels = 16
        self.conv = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(
            self.conv.weight, mode='fan_out', nonlinearity='relu'
        )
        self.bn = nn.BatchNorm2d(16)
        self.layer1 = self.make_layer(16, 1)
        self.layer2 = self.make_layer(32, 2)
        self.layer3 = self.make_layer(64, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)
        nn.init.kaiming_normal_(
            self.fc.weight, mode='fan_out', nonlinearity='relu'
        )
    
    def make_layer(self, out_channels, stride):
        subsample = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if out_channels != self.in_channels else None
        if subsample:
            nn.init.kaiming_normal_(
                subsample[0].weight, mode='fan_out', nonlinearity='relu'
            )
        layers = [ResBlock(self.in_channels, out_channels, stride, subsample)]
        self.in_channels = out_channels
        for _ in range(1, self.N):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x = F.relu(self.bn(self.conv(x)))
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

        
train_loader = DataLoader(
    datasets.CIFAR10('data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ) # 注意input image是三通道的，所以mean和std需要三维
        ])),
    batch_size=128, shuffle=True, drop_last=False,
)
test_loader = DataLoader(
    datasets.CIFAR10('data', train=False, download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ) # 注意input image是三通道的，所以mean和std需要三维
        ])),
    batch_size=128, shuffle=True, drop_last=False,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
iter_loss = []
batch_loss = []
avg_list = []
acc_list = []
epoN = 200
model = MyResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9
)
# 首先用0.01的lr进行warmup，正确率>20%时转为0.1，然后
# 在iteration = 32k、48k时分别在原lr的基础上乘以0.1，逐步下降
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda iter: 
        0.1 if iter < 500 else 
        1 if iter < 32000 else 
        0.1 if iter < 48000 else 0.01
) 

for epoch in range(epoN):
    model.train()
    progress_bar = tqdm(train_loader, total=len(train_loader)) 
    cor, tot = 0, 0
    for x, y in progress_bar:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        batch_loss.append(loss.data.cpu().numpy())
        loss.cpu().backward()
        optimizer.step()
        scheduler.step()
        cor += (pred.max(1)[1] == y).sum().item() 
        tot += x.shape[0]
        progress_bar.set_description('Train | epoch [{}/{}] | loss: {:.3f} | acc: {:.3f}'.format(epoch+1, epoN, loss.item(), cor/tot))

    iter_loss.append(np.average(np.array(batch_loss)))
        
    model.eval()
    total_loss = 0 
    cor, tot = 0, 0
    progress_bar = tqdm(test_loader, total =len(test_loader))
    for x, y in progress_bar:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():  #禁止计算梯度，只计算output(速度加快)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.cpu().item()*len(x)
            cor += (pred.max(1)[1] == y).sum().item()
            tot += x.shape[0]
            progress_bar.set_description('Test  | epoch [{}/{}] | loss: {:.3f} | acc: {:.3f}'.format(epoch+1, epoN, loss.item(), cor/tot))
            
    avg_loss = total_loss / len(test_loader.dataset)
    avg_list.append(avg_loss)
    acc_list.append(cor/tot)

# 最后将模型保存起来
torch.save(model.state_dict(), "model/resnet_param.pkl")
