# 生成对抗网络的音频处理
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader,Dataset
import os
import torch.optim as optim

file=os.listdir('D:\QQ音乐\新建文件夹')
path=[os.path.join('D:\QQ音乐\新建文件夹',f) for f in file]

# (1)处理数据集
class audiodataset(Dataset):
    def __init__(self,audiofile):
        self.file=audiofile
        self.transforms=torchaudio.transforms.MelSpectrogram()
    def __len__(self):
        return len(self.file)
    def __getitem__(self, idx):
        audio,sr=torchaudio.load(self.file[idx])
        audio = audio / audio.abs().max()
        audio=self.transforms(audio)
        return audio

# (2)判别器
class audiodiscriminor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(2,3,kernel_size=3,padding=1),
                                nn.LeakyReLU(0.2),
                                nn.Conv2d(3,4,3,padding=1),
                                nn.LeakyReLU(0.2),
                                nn.Conv2d(4,5,3,padding=1),
                                nn.BatchNorm2d(5),
                                nn.LeakyReLU(0.2),
                                nn.Flatten()
                                )
        self.fc=nn.Sequential(nn.Linear(5*128*662,1),
                            nn.Sigmoid())
    def forward(self,x):
        x=self.conv(x)
        x=self.fc(x)
        return x

# (3)生成器
class audiogenerator(nn.Module):
    def __init__(self,noise_dim,outchannels):
        super().__init__()
        self.fc=nn.Sequential(nn.Linear(noise_dim,5*128*662),
                              nn.ReLU(True))
        self.conv=nn.Sequential(nn.Conv2d(5,4,3,padding=1),
                                nn.BatchNorm2d(4),
                                nn.LeakyReLU(0.2),
                                nn.Conv2d(4,3,3,padding=1),
                                nn.BatchNorm2d(3),
                                nn.ReLU(True),
                                nn.Conv2d(3,out_channels=outchannels,kernel_size=3,padding=1),
                                nn.BatchNorm2d(outchannels),
                                nn.Tanh()
                                )
    def forward(self,x):
            x=self.fc(x)
            x=x.view(-1,5,128,662)
            x=self.conv(x)
            return x

# (4)其它参数及优化器
noisedim=100
outchannels=2
batch_size=3
G=audiogenerator(noisedim,outchannels)
D=audiodiscriminor()
G_optimizer=optim.Adam(G.parameters(),weight_decay=0.0001)
D_optimizer=optim.Adam(D.parameters(),weight_decay=0.0001)
criterion=nn.BCELoss()
dataset=audiodataset(audiofile=path)
dataloader=DataLoader(dataset,batch_size=16,shuffle=True)

for _,data in enumerate(dataloader):
    print(data.shape)

# (5)训练
def train(epochs):
    for epoch in range(epochs):
        for _,data in enumerate(dataloader):

            # 训练判别器
            zaosheng=torch.randn(batch_size,noisedim)
            fake_data=G(zaosheng)
            dis_real=D(fake_data).view(-1)
            lossA=criterion(dis_real,torch.zeros_like(dis_real))
            is_real=D(data).view(-1)
            lossB=criterion(is_real,torch.ones_like(is_real))
            lossC=lossA+lossB
            D_optimizer.zero_grad()
            lossC.backward()
            D_optimizer.step()
            # 训练生成器

            fake_data=G(zaosheng)
            dis_real=D(fake_data).view(-1)
            lossD=criterion(dis_real,torch.ones_like(dis_real))
            G_optimizer.zero_grad()
            lossD.backward()
            G_optimizer.step()
        print(f'epoch/epochs:{epoch}/{epochs},LossC:{lossC},LossD:{lossD}')

train(1)
