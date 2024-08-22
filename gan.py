import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class D(nn.Module):
    def __init__(self,img_size):
        super().__init__()
        self.d=nn.Sequential(
            nn.Conv2d(1,3,3,padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(img_size*3,128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.d(x)

class G(nn.Module):
    def __init__(self,zao_size,img_size):
        super().__init__()
        self.g=nn.Sequential(
            nn.Linear(zao_size,128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,img_size),
            nn.Tanh()
        )
    def forward(self,x):
        return self.g(x).view(-1,1,28,28)

lr=0.0001
epochs=5
batch_size=16
zao_size=32
img_size=28*28
panbieqi=D(img_size)
shengchengqi=G(zao_size,img_size)
optimizer_pan=optim.Adam(panbieqi.parameters(),lr=lr,weight_decay=0.0001)
optimizer_sheng=optim.Adam(shengchengqi.parameters(),lr=lr,weight_decay=0.0001)
zaosheng=torch.randn(size=(batch_size,zao_size))
transforms=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.3),(0.3))])
dataset=datasets.MNIST(root='data/',transform=transforms)
dataloader=DataLoader(dataset,batch_size=16,shuffle=True)
criterion=nn.BCELoss()

a=torch.randn(size=(16,32))
b=shengchengqi(a)
print(b.shape)
c=panbieqi(b)
print(c.shape)

def train(epochs):
    for epoch in range(epochs):
        for idx,(real,_) in enumerate(dataloader):
            # 训练生成器
            zaosheng=torch.randn(size=(batch_size,zao_size))
            fake=shengchengqi(zaosheng)
            dis_real=panbieqi(fake).view(-1)
            lossA=criterion(dis_real,torch.zeros_like(dis_real))
            is_real=panbieqi(real).view(-1)
            lossB=criterion(is_real,torch.ones_like(is_real))
            lossC=(lossA+lossB)/2
            optimizer_pan.zero_grad()
            lossC.backward(retain_graph=True)
            optimizer_pan.step()

            # 训练判别器
            output=panbieqi(fake).view(-1)
            lossD=criterion(output,torch.ones_like(output))
            optimizer_sheng.zero_grad()
            lossD.backward(retain_graph=True)
            optimizer_sheng.step()
        print(f'epoch/epochs:{epoch}/{epochs},Loss C：{lossC},Loss D:{lossD}')
    torch.save(shengchengqi.state_dict(),'shengchengqi.pth')
# 训练
# train(4)

# 加载训练模型
shengchengqi.load_state_dict(torch.load('shengchengqi.pth'))
zaosheng=torch.randn(size=(batch_size,zao_size))
with torch.no_grad():
    fake_images=shengchengqi(zaosheng).view(-1,1,28,28)
fakeimage = fake_images
print(fakeimage.shape)

import matplotlib.pyplot as plt
fig,axs=plt.subplots(4,4,figsize=(8,8))
for i in range(4):
    for j in range(4):
        axs[i,j].imshow(fakeimage[i*4+j].squeeze().numpy(),cmap='cool')
        axs[i, j].axis('off')
plt.show()
