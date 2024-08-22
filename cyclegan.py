#  CycleGan
import torch
import torch.nn as nn

# (1)Block
class Block(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=stride,padding=1,bias=True,padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2,True),)
    def forward(self,x):
        return self.conv(x)

# (2)判别器 Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels=3
        self.features=[64,128,256,512]
        self.init=nn.Sequential(
            nn.Conv2d(self.in_channels,self.features[0],kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.InstanceNorm2d(self.features[0]),
            nn.LeakyReLU(0.2),)
        layers=[]
        in_channels=self.features[0]
        for i in self.features[1:]:
            layers.append(Block(in_channels=in_channels,out_channels=i,stride=1))
            in_channels=i
        layers.append(nn.Conv2d(512,1,kernel_size=4,stride=1,padding=1,padding_mode='reflect'))
        self.model=nn.Sequential(*layers)
    def forward(self,x):
        x=self.init(x)
        return self.model(x)

# (3)可配置的卷积块 下采样和上采样
class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,down=True,use_act=True,**kwargs):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(in_channels,out_channels,padding_mode='reflect',**kwargs)
                  if down
                  else nn.ConvTranspose2d(in_channels,out_channels,**kwargs),
                  nn.InstanceNorm2d(out_channels),
                  nn.LeakyReLU(0.2) if use_act else nn.Identity(),)
    def forward(self,x):
        return self.conv(x)

# (4)残差模块
class ResidualBlock(nn.Module):
    def __init__(self,in_channels):
        super(ResidualBlock, self).__init__()
        self.block=nn.Sequential(ConvBlock(in_channels,in_channels,kernel_size=3, padding=1),
                                 ConvBlock(in_channels,in_channels,use_act=False,kernel_size=3, padding=1),)
    def forward(self,x):
        return x+self.block(x)

# (5)生成器
class Generator(nn.Module):
    def __init__(self,img_channels,num_features=64,num_residuals=9):
        super().__init__()
        self.initial=nn.Sequential(
            nn.Conv2d(img_channels,num_features,kernel_size=7,stride=1,padding=3,padding_mode='reflect'),
            nn.InstanceNorm2d(num_features),
            nn.LeakyReLU(0.2,inplace=True),)
        self.down_blocks=nn.ModuleList(
                                    [ConvBlock(num_features,num_features*2,
                                                 kernel_size=3,stride=2,padding=1),
                                    ConvBlock(num_features*2,num_features*4,
                                                 kernel_size=3,stride=2,padding=1),
                                    nn.Conv2d(num_features*4,num_features*4,kernel_size=3,padding=1)])
        self.res_blocks=nn.ModuleList([ResidualBlock(num_features*4) for _ in range(num_residuals)])
        self.up_blocks=nn.ModuleList([ConvBlock(num_features*4,num_features*2,down=False,
                                                kernel_size=3,stride=2,padding=1,output_padding=1),
                                      ConvBlock(num_features*2,num_features,down=False,kernel_size=3,
                                                stride=2,padding=1,output_padding=1),
                                      nn.Conv2d(num_features,num_features,kernel_size=3,padding=1)])
        self.last=nn.Conv2d(num_features,img_channels,kernel_size=7,stride=1,padding=3,
                            padding_mode='reflect')
        self.act= nn.Tanh()
    def forward(self,x):
        x=self.initial(x)
        for layer in self.down_blocks:
            x=layer(x)
        for res in self.res_blocks:
            x=res(x)
        for up in self.up_blocks:
            x=up(x)
        x=self.last(x)
        return self.act(x)

# (6)数据集处理
from PIL import Image
import os
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms

Transforms=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=(100,100)),transforms.Normalize((0.3),(0.3)),transforms.RandomHorizontalFlip(p=0.5)])

class HorseZebraDataset(Dataset):
    def __init__(self,root_zebra,root_horse,transforms=None):
        super().__init__()
        self.root_zebra=root_zebra
        self.root_horse=root_horse
        self.transforms=transforms

        self.zebra_image=os.listdir(root_zebra)
        self.horse_image=os.listdir(root_horse)
        self.len_dataset=max(len(self.zebra_image),len(self.horse_image))
        self.zebra_len=len(self.zebra_image)
        self.horse_len=len(self.horse_image)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        zebra_image = self.zebra_image[index % self.zebra_len]
        horse_image = self.horse_image[index % self.horse_len]

        zebra_path=os.path.join(self.root_zebra,zebra_image)
        horse_path=os.path.join(self.root_horse,horse_image)

        zebra_img=np.array(Image.open(zebra_path).convert('RGB'))
        horse_img=np.array(Image.open(horse_path).convert('RGB'))

        if self.transforms:
            zebra_img=self.transforms(zebra_img)
            horse_img=self.transforms(horse_img)

        return zebra_img,horse_img



# (7)训练
import torch.optim as optim

datasets=HorseZebraDataset(root_zebra='DALI/',root_horse='DALI/',transforms=Transforms)
dataloader=DataLoader(dataset=datasets,batch_size=1,shuffle=True)
img_channels=3

G_zebra=Generator(img_channels)
G_horse=Generator(img_channels)
D_zebra=Discriminator()
D_horse=Discriminator()

optimizer_G_zebra=optim.Adam(G.parameters(),lr=0.0001,weight_decay=0.001)
optimizer_G_horse=optim.Adam(G.parameters(),lr=0.0001,weight_decay=0.001)
optimizer_D_zebra=optim.Adam(D.parameters(),lr=0.0001,weight_decay=0.002)
optimizer_D_horse=optim.Adam(D.parameters(),lr=0.0001,weight_decay=0.002)

L1=nn.L1Loss()
criterion=nn.MSELoss()

def train(epochs):
    for epoch in range(epochs):
        for _,(zebra,horse) in enumerate(dataloader):
            # (1)训练判别器 D_horse
            
            fake_horse=G_horse(zebra)
            D_horse_real=D_horse(horse)
            D_horse_fake=D_horse(fake_horse.detach())

            D_H_real_loss=criterion(D_horse_real,torch.ones_like(D_horse_real))
            D_H_fake_loss=criterion(D_horse_fake,torch.zeros_like(D_horse_fake))

            D_H_loss=(D_H_real_loss+D_H_fake_loss)/2

            # (2)训练判别器 D_zebra
            fake_zebra=G_zebra(horse)
            D_zebra_real=D_zebra(zebra)
            D_zebra_fake=D_zebra(fake_zebra.detach())

            D_Z_real_loss=criterion(D_zebra_real,torch.ones_like(D_zebra_real))
            D_Z_fake_loss=criterion(D_zebra_fake,torch.zeros_like(D_zebra_fake))

            D_Z_loss=(D_Z_fake_loss+D_Z_real_loss)/2

            # (3)总的判别器的 loss
            D_loss=(D_Z_loss+D_H_loss)/2
            optimizer_D_horse.zero_grad()
            optimizer_D_zebra.zero_grad()
            D_loss.backward()
            optimizer_D_horse.step()
            optimizer_D_zebra.step()

            # (4)训练生成器 G_horse 和 G_zebra
            fake_horse=G_horse(zebra)
            D_horse_fake=D_horse(fake_horse.detach())
            fake_zebra=G_zebra(horse)
            D_zebra_fake=D_zebra(fake_zebra.detach())

            loss_G_horse=criterion(D_horse_fake,torch.ones_like(D_horse_fake))
            loss_G_zebra=criterion(D_zebra_fake,torch.ones_like(D_zebra_fake))

            # (5)循环一致性损失计算 cycleloss
            cycle_zebra=G_zebra(fake_horse)
            cycle_horse=G_horse(fake_zebra)
            cycle_zebra_loss=L1(zebra,cycle_zebra)
            cycle_horse_loss=L1(horse,cycle_horse)

            # (6)身份损失（identity loss）为了帮助生成器更好地保留输入图像的特征，防止生成器产生不必要的修改
            identity_zebra=G_zebra(zebra)
            identity_horse=G_horse(horse)
            identity_zebra_loss=L1(zebra,identity_zebra)
            identity_horse_loss=L1(horse,identity_horse)
            # (7) 生成器的总损失
            G_loss=(loss_G_zebra+loss_G_horse
                    +cycle_zebra_loss+cycle_horse_loss
                    +identity_zebra_loss+identity_horse_loss)/6

            optimizer_G_zebra.zero_grad()
            optimizer_G_horse.zero_grad()
            G_loss.backward()
            optimizer_G_zebra.step()
            optimizer_G_horse.step()
        print(f'epoch/epochs:{epoch}/{epochs},LossD:{D_loss},LossG:{G_loss}')
    # torch.save(G_zebra.state_dict(),'G_zebra.pth')
    # torch.save(G_horse.state_dict(),'G_horse.pth')
    # torch.save(D_zebra.state_dict(),'D_zebra.pth')
    # torch.save(D_horse.state_dict(),'D_horse.pth')

# train(1)

model=G_zebra
model.load_state_dict(torch.load('G_zebra.pth'))
