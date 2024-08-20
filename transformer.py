import torch
import torch.nn as nn
import math
import numpy as np
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader,Dataset
device="cuda"
sentence=[['我有一个好朋友P','S I have a good friend . ','I have a good friend . E'],['我有零个女朋友P', 'S I have zero girl friend . ', 'I have zero girl friend . E'],['我有一个男朋友P', 'S I have a boy friend . ', 'I have a boy friend . E']]
srv_vocab={'P':0,'我':1,'有':2,'一':3,'个':4,'好':5,'朋':6,'友':7,'零':8,'女':9,'男':10}
src_idx2word={i:w for i,w in enumerate(srv_vocab)}
src_vocab_size=len(srv_vocab)

tgt_vocab={'P':0,'I':1,'have':2,'a':3,'good':4,'friend':5,'zero':6,'girl':7,'boy':8,'S':9,'E':10,'.':11}
tgt_idx2word={i:w for i,w in enumerate(tgt_vocab)}
tgt_vocab_size=len(tgt_vocab)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

src_len=8
tgt_len=7
# 编码的维度
d_model=512
d_ff=2048
# Q和K的维度
d_k=d_v=64
n_layers=6
n_head=8

# 数据处理
def make_data(sentence):
    enc_inputs,dec_inputs,dec_outs=[],[],[]
    for i in range(len(sentence)):
        enc_input=[[srv_vocab[n] for n in sentence[i][0]]]
        dec_input=[[tgt_vocab[n] for n in sentence[i][1].split()]]
        dec_out=[[tgt_vocab[n] for n in sentence[i][2].split()]]
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outs.extend(dec_out)
    return torch.tensor(enc_inputs,dtype=torch.long),\
           torch.tensor(dec_inputs,dtype=torch.long),\
           torch.tensor(dec_outs,dtype=torch.long)

enc_inputs,dec_inputs,dec_outputs = make_data(sentence)

# 构建数据传入Dataloader
class Mydataset(Dataset):
    def __init__(self,enc_inputs,dec_inputs,dec_outs):
        super().__init__()
        self.enc_inputs=enc_inputs
        self.dec_inputs=dec_inputs
        self.dec_outs=dec_outs

    def __len__(self):
        return len(enc_inputs)

    def __getitem__(self, idx):
        return self.enc_inputs[idx],self.dec_inputs[idx],self.dec_outs[idx]

data=Mydataset(enc_inputs,dec_inputs,dec_outputs)

# 构建Dataloader
dataloader=DataLoader(dataset=data,batch_size=2,shuffle=True)

# 构建Transformer模型
# （1）位置编码
class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model=512
        self.dropout_parameters=0.1
        self.max_len=5000
        self.dropout=nn.Dropout(p=self.dropout_parameters)
        pe=torch.zeros(self.max_len,self.d_model)
        position=torch.arange(0,self.max_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,self.d_model,2).float()*(-math.log(10000.0)/self.d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe',pe)

    def forward(self,x):
            """
            x: [seq_len, batch_size, d_model]
           """
            x=x+self.pe[0:x.shape[0],:]
            return self.dropout(x)

# (2)生成注意力机制的掩码
def get_attention_pad_mask(seq_q,seq_k):
    batch_size,len_q=seq_q.size()
    batch_size,len_k=seq_k.size()
    pad_attention_mask=seq_k.data.eq(0).unsqueeze(1)
    return pad_attention_mask.expand(batch_size,len_q,len_k)

# (3)生成用于掩盖未来信息的掩码（subsequence mask）
def get_attn_subsequence_mask(seq):
    """打印出来看看是什么的输出（一目了然）
    seq: [batch_size, tgt_len]
    """
    attn_shape=[seq.size(0),seq.size(1),seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask=np.triu(np.ones(attn_shape),k=1)
    subsequence_mask=torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask

# (4)点积注意力机制
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,Q,K,V,attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        scores=torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(d_k)
        # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask,-1e9)
        softmax=nn.Softmax(dim=-1)
        attn=softmax(scores)
        context=torch.matmul(attn,V)
        return  context,attn

# (5)实现多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.d_model=512
        self.d_k=64
        self.d_v=64
        self.n_head=8
        self.W_Q=nn.Linear(self.d_model,self.d_k*self.n_head,bias=False)
        self.W_K=nn.Linear(self.d_model,self.d_k*self.n_head,bias=False)
        self.W_V=nn.Linear(self.d_model,self.d_k*self.n_head,bias=False)
        self.fc=nn.Linear(self.d_k*self.n_head,self.d_model,bias=False)
    def forward(self,input_Q,input_K,input_V,attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual,batch_size=input_Q, input_Q.size(0)
        Q=self.W_Q(input_Q).view(batch_size,-1,self.n_head,d_k).transpose(1,2)
        # Q: [batch_size, n_heads, len_q, d_k]
        K=self.W_K(input_K).view(batch_size,-1,self.n_head,d_k).transpose(1,2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        V=self.W_V(input_V).view(batch_size,-1,self.n_head,d_v).transpose(1,2)
        attn_mask=attn_mask.unsqueeze(1).repeat(1,self.n_head,1,1)
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        context,attn=ScaleDotProductAttention()(Q,K,V,attn_mask)
        # context: [batch_size, n_heads, len_q, d_v],
        # attn: [batch_size, n_heads, len_q, len_k]
        context=context.transpose(1,2).reshape(batch_size,-1,self.n_head*self.d_v)
        output=self.fc(context)
        # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model)(output+residual),attn

# （6）位置前馈网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet,self).__init__()
        self.d_model=512
        self.d_ff=2048
        self.fc=nn.Sequential(nn.Linear(self.d_model,self.d_ff,bias=False),
                              nn.ReLU(),
                              nn.Linear(self.d_ff,self.d_model,bias=False)
        )
    def forward(self,input):
        residual=input
        out=self.fc(input)
        return nn.LayerNorm(d_model)(residual+out)

# （7）Encoder Layer层
class Encoderlayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_self_attn=MultiHeadAttention()
        self.pos_ffn=PoswiseFeedForwardNet()
    def forward(self,enc_inputs,enc_self_attn_mask):
        """
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        enc_outputs,attn =self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)
        enc_outputs=self.pos_ffn(enc_outputs)
        return enc_outputs,attn

# （7）Decoder Layer层
class Decoderlayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_self_attn=MultiHeadAttention()
        self.dec_enc_attn=MultiHeadAttention()
        self.pos_ffn=PoswiseFeedForwardNet()
    def forward(self,dec_inputs,enc_outputs,dec_self_attn_mask,dec_enc_attn_mask):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        dec_outputs,dec_self_attn=self.dec_self_attn(dec_inputs,dec_inputs,dec_inputs,dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model],
        # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs,dec_enc_attn=self.dec_enc_attn(dec_outputs,enc_outputs,enc_outputs,dec_enc_attn_mask)
         # dec_outputs: [batch_size, tgt_len, d_model],
        # dec_enc_attn: [batch_size, n_heads, tgt_len, src_len]
        dec_outputs=self.pos_ffn(dec_outputs)
        # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs,dec_self_attn,dec_enc_attn
#（8）Encoder实现
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model=512
        self.src_vocab_size=11
        self.n_layers=6
        self.src_emb=nn.Embedding(self.src_vocab_size,self.d_model)# token Embedding
        self.pos_emb=PositionalEncoding()# Transformer中位置编码时固定的，不需要学习
        self.layers=nn.ModuleList(modules=(Encoderlayer() for _ in range(self.n_layers)))
    def forward(self,enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """
        enc_outputs=self.src_emb(enc_inputs)
        # [batch_size, src_len, d_model]
        enc_outputs=self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1)
        # [batch_size, src_len, d_model]
        enc_self_attn_mask=get_attention_pad_mask(enc_inputs,enc_inputs)
        # [batch_size, src_len, src_len]
        enc_self_attns=[] # 用于存储每一层的自注意力权重
        for layer in self.layers:
            enc_outputs,enc_self_attn=layer(enc_outputs,enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)# 保存自注意力权重（主要用于可视化）
        return enc_outputs,enc_self_attns
#（9）Decoder实现
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model=512
        self.tgt_vocab_size=12
        self.n_layers=6
        self.tgt_emb=nn.Embedding(self.tgt_vocab_size,self.d_model)# 目标词表的嵌入层
        self.pos_emb=PositionalEncoding()
        self.layers=nn.ModuleList([Decoderlayer() for _ in range(self.n_layers)])# 解码器层
    def forward(self,dec_inputs,enc_inputs,enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_inputs: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]  # 来自编码器的输出
        """
        dec_outputs=self.tgt_emb(dec_inputs)
        #目标词嵌入: [batch_size, tgt_len, d_model]
        dec_outputs=self.pos_emb(dec_outputs.transpose(0,1)).transpose(0,1).to(device)
        # 添加位置编码: [batch_size, tgt_len, d_model]
        # 计算 Decoder 的 self-attention 的 mask

        dec_self_attn_pad_mask=get_attention_pad_mask(dec_inputs,dec_inputs).to(device)
         # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask=get_attn_subsequence_mask(dec_inputs).to(device)
         # [batch_size, tgt_len, tgt_len]  Masked Self_Attention：当前时刻是看不到未来的信息的

        dec_self_attn_mask=torch.gt((dec_self_attn_pad_mask+dec_self_attn_subsequence_mask),0)\
            .to(device) # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）
        # [batch_size, tgt_len, tgt_len]; torch.gt比较两个矩阵的元素，大于则返回1，否则返回0

        dec_enc_attn_mask=get_attention_pad_mask(dec_inputs,enc_inputs)
        # [batc_size, tgt_len, src_len]

        dec_self_attns,dec_enc_attns=[],[]
        # dec_outputs: [batch_size, tgt_len, d_model],
        # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len],
        # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        for layer in self.layers:
            dec_outputs,dec_self_attn,dec_enc_attn=layer(dec_outputs,enc_outputs,dec_self_attn_mask,dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs,dec_self_attns,dec_enc_attns
        # dec_outputs: [batch_size, tgt_len, d_model]

#（10）Transformer实现
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model=512
        self.tgt_vocab_size=12
        self.encoder=Encoder().to(device)
        self.decoder=Decoder().to(device)
        self.projection=nn.Linear(self.d_model,self.tgt_vocab_size,bias=False).to(device)
    def forward(self,enc_inputs,dec_inputs):
        """
        enc_inputs: [batch_size, src_len]  # 输入序列（源语言）
        dec_inputs: [batch_size, tgt_len]  # 输入序列（目标语言）
        """
        enc_outputs,enc_self_attns=self.encoder(enc_inputs)
        # 编码器的前向传播，生成编码器的输出和自注意力权重
        # enc_outputs: [batch_size, src_len, d_model],
        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]

        dec_outputs,dec_self_attns,dec_enc_attns=self.decoder(dec_inputs,enc_inputs,enc_outputs)
        # 解码器的前向传播，生成解码器的输出、自注意力权重和编码器-解码器注意力权重
        # dec_outputs: [batch_size, tgt_len, d_model],
        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
        # dec_enc_attns: [n_layers, batch_size, tgt_len, src_len]

        dec_logits=self.projection(dec_outputs)
        # 投影层，将解码器的输出映射到目标词汇表的大小
        # dec_outputs: [batch_size, tgt_len, d_model]
        # -> dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1,dec_logits.size(-1)),\
               enc_self_attns,\
               dec_self_attns,\
               dec_enc_attns
        # 将输出调整为二维张量（batch_size * tgt_len, tgt_vocab_size）

#（11）训练
model=Transformer().to(device)
criterion=nn.CrossEntropyLoss(ignore_index=0)
optimizer=optim.SGD(model.parameters(),lr=0.0001,weight_decay=0.001)

def train(epoches):
    for epoch in range(epoches):
        for enc_inputs,dec_inputs,dec_outputs in dataloader:
            # enc_inputs: [batch_size, src_len]
            # dec_inputs: [batch_size, tgt_len]
            # dec_outputs: [batch_size, tgt_len]

            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), \
                                                  dec_inputs.to(device), \
                                                  dec_outputs.to(device) # 将数据加载到GPU或者CPU上

            outputs,enc_self_attns,dec_self_attns,dec_enc_attns=model(enc_inputs,dec_inputs)
             # outputs: [batch_size * tgt_len, tgt_vocab_size]

            loss=criterion(outputs,dec_outputs.view(-1))
            # dec_outputs.view(-1):[batch_size * tgt_len * tgt_vocab_size]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch:','%04d' % (epoch+1),'loss=',f'{loss}')
