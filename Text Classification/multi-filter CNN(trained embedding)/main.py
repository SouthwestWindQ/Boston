import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from torchvision import datasets, transforms
from tqdm import tqdm   #python进度条！
import jieba

dic = {}    #训练、验证、测试集中的所有句子中的词汇/空格/数字等

class textCNN(nn.Module):
    def __init__(self):
        super(textCNN,self).__init__()
        self.embedding = nn.Embedding(len(dic),128,padding_idx=dic['<PAD>'])
        self.conv1 = nn.Conv2d(1,16,(3,128))
        self.conv2 = nn.Conv2d(1,16,(4,128))
        self.conv3 = nn.Conv2d(1,16,(5,128))
        self.fc1 = nn.Linear(48,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,4)
    def forward(self,x):
        x = self.embedding(x).cuda()
        x = x.unsqueeze(1)
        conv1 = F.relu(self.conv1(x)).squeeze(-1)
        conv2 = F.relu(self.conv2(x)).squeeze(-1)
        conv3 = F.relu(self.conv3(x)).squeeze(-1)
        pool1 = F.max_pool1d(conv1,18).squeeze(-1)
        pool2 = F.max_pool1d(conv1,17).squeeze(-1)
        pool3 = F.max_pool1d(conv1,16).squeeze(-1)
        x = torch.cat((pool1,pool2,pool3),dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def ToSentence(words):
    sentences = []
    tmp = 0
    for i in range(len(words)):
        if words[i]=='\t':
            sentence = []
            for j in range(tmp,i):
                sentence.append(words[j])
            if sentence[0]=='\n':
                sentence.remove('\n')
            sentences.append(sentence)
            tmp = i+2
    return sentences

def ExtractLabel(words):
    labels = []
    for i in range(len(words)):
        if words[i]=='\t':
            labels.append(int(words[i+1]))
    return torch.tensor(labels)
            
def coding(sentences,maxlen):
    code_list = []
    for sentence in sentences:
        codes = []
        for word in sentence:
            codes.append(dic[word])
        if len(codes) != maxlen:
            for i in range(maxlen-len(codes)):
                codes.append(dic['<PAD>'])
        code_list.append(codes)
    return torch.tensor(code_list)

train_data = open("../data/train.txt","r", encoding='utf-8').read()
dev_data = open("../data/dev.txt","r", encoding='utf-8').read()
test_data = open("../data/test.txt","r", encoding='utf-8').read()
train_words = jieba.lcut(train_data)
dev_words = jieba.lcut(dev_data)
test_words = jieba.lcut(test_data)
all_words = train_words + dev_words + test_words
for i in range(len(all_words)):
    if not(all_words[i]=='\t' or all_words[i-1]=='\t'):
        dic[all_words[i]] = 0
dic['<PAD>'] = 0
sum = 0
for word in dic.keys():
    dic[word] = sum
    sum += 1
train_sentence = ToSentence(train_words) #最长句子：20
dev_sentence = ToSentence(dev_words)     #最长句子：17
test_sentence = ToSentence(test_words)   #最长句子：18
train_label = ExtractLabel(train_words)
dev_label = ExtractLabel(dev_words)
test_label = ExtractLabel(test_words)
train_coding = coding(train_sentence,maxlen=20)
dev_coding = coding(dev_sentence,maxlen=20)
test_coding = coding(test_sentence,maxlen=20)
train_set = TensorDataset(train_coding,train_label)
dev_set = TensorDataset(dev_coding,dev_label)
test_set = TensorDataset(test_coding,test_label)
train_loader = DataLoader(dataset=train_set, batch_size=50, shuffle=True, drop_last=False)
dev_loader = DataLoader(dataset=dev_set, batch_size=50, shuffle=False, drop_last=False)
test_loader = DataLoader(dataset=test_set, batch_size=50, shuffle=False, drop_last=False)

iter_loss = []
dev_loss = []
acc_list = []
epoN = 200
model = textCNN().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
writer = SummaryWriter(log_dir="./logs/",flush_secs=60)
for epoch in range(epoN):
    model.train()
    progress_bar = tqdm(train_loader, total=len(train_loader)) #进度条，使进度的显示更清晰，学习！
    cor,tot,batch_loss = 0,0,[]
    for x,y in progress_bar:
        optimizer.zero_grad()
        x,y = x.cuda(),y.cuda()
        pred = model(x)
        loss = criterion(pred, y)
        batch_loss.append(loss.data.cpu().numpy())
        loss.backward(retain_graph=True)
        optimizer.step()
        cor += (pred.max(1)[1] == y).sum().item()  #学习这种简省的写法！
        tot += x.shape[0]
        progress_bar.set_description(
            'Train | epoch [{}/{}] | loss: {:.3f} | acc: {:.3f}'.format(
                epoch+1, epoN, np.average(np.array(batch_loss)), cor/tot
            )
        )

    iter_loss.append(np.average(np.array(batch_loss)))

    model.eval()
    cor,tot,batch_loss = 0,0,[]
    progress_bar = tqdm(dev_loader, total=len(dev_loader))
    for x,y in progress_bar:
        x,y = x.cuda(), y.cuda()
        with torch.no_grad():  #禁止计算梯度，只计算output(速度加快)
            pred = model(x)
            loss = criterion(pred, y)
            batch_loss.append(loss.data.cpu().numpy())
            cor += (pred.max(1)[1] == y).sum().item()
            tot += x.shape[0]
            progress_bar.set_description(
                'Dev   | epoch [{}/{}] | loss: {:.3f} | acc: {:.3f}'.format(
                    epoch+1, epoN, np.average(np.array(batch_loss)), cor/tot
                )
            )
    dev_loss.append(np.average(np.array(batch_loss)))
    acc_list.append(cor/tot)
    writer.add_scalar('Accuracy',cor/tot,epoch+1)

writer.close()
# plt.scatter(x,y2)
# plt.show()

# x = np.array([i for i in range(1,epoN+1)])
# y1 = np.array(iter_loss)
# y2 = np.array(dev_loss)
# plt.scatter(x,y1)
# plt.scatter(x,y2)
# plt.show()

model.eval()
cor,tot,batch_loss = 0,0,[]
progress_bar = tqdm(test_loader, total=len(test_loader))
for x,y in progress_bar:
    x,y = x.cuda(), y.cuda()
    with torch.no_grad():  #禁止计算梯度，只计算output(速度加快)
        pred = model(x)
        loss = criterion(pred, y)
        batch_loss.append(loss.data.cpu().numpy())
        cor += (pred.max(1)[1] == y).sum().item()
        tot += x.shape[0]
        progress_bar.set_description(
            'Test  | loss: {:.3f} | acc: {:.3f}'.format(
                np.average(np.array(batch_loss)), cor/tot
            )
        )
print(cor,tot,cor/tot)
