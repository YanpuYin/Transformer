# LoRA微调原理
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class LoRA(nn.Module):
    def __init__(self,in_features,out_features,merge,rank=16,lora_alpha=16,dropout=0.5):
        super.__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.merge=merge
        self.rank=rank
        self.lora_alpha=lora_alpha
        self.dropout_rate=dropout
        self.linear=nn.Linear(in_features,out_features)
        if rank>0:
            self.lora_b=nn.Parameter(torch.zeros(self.rank,self.out_features))
            self.lora_a=nn.Parameter(torch.zeros(self.in_features,self.rank))
            self.scale=self.lora_alpha/self.rank
            self.linear.weight.requires_grad=False

        if self.dropout_rate>0:
            self.dropout=nn.Dropout(self.dropout_rate)
        else:
            self.dropout=nn.Identity()

        self.initialize()

    def initialize(self):
        nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self,x):

        if self.rank>0 and self.merge:
            output=F.linear(x,self.Linear.weight+self.lora_a @ self.lora_b * self.scale,self.linear.bias)
            output=self.dropout(output)
            return output

        else:
            output=self.linear(x)
            output=self.dropout(output)
            return output
