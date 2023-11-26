# -*- encoding: utf-8 -*-
"""

@File    :   model.py  
@Modify Time : 2023/11/19 16:38 
@Author  :  Allen.Yang  
@Contact :   MC36514@um.edu.mo        
@Description  : 主模型

"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Seq2Seq 主调用模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, left_hidden, right_hidden,tgt,is_first):
        encoder_output = self.encoder(left_hidden,right_hidden,is_first)
        output,hidden = self.decoder(tgt,encoder_output,encoder_output)
        return output,hidden

# 编码器
class Encoder(nn.Module):
    def __init__(self, enb_dim,vocab_dim,hidden_dim):
        super(Encoder, self).__init__()
        self.embeddingl = nn.Embedding(vocab_dim, enb_dim)
        self.embeddingr = nn.Embedding(vocab_dim, enb_dim)
        self.wl_linear = nn.Linear(enb_dim,enb_dim)
        self.wr_linear = nn.Linear(enb_dim,enb_dim)
        self.wl_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.wr_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.gl_linear = nn.Linear(enb_dim,3)
        self.gr_linear = nn.Linear(enb_dim,3)
        self.gl_linear1 = nn.Linear(hidden_dim, 3)
        self.gr_linear2 = nn.Linear(hidden_dim, 3)
        self.linearout = nn.Linear(enb_dim,hidden_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, left_hidden, right_hidden,is_first):
        # 第一轮走这边
        if is_first:
            bias = self.gl_linear(left_hidden) + self.gr_linear(right_hidden)
            bias = F.softmax(bias, dim=1)
            h_tem = self.sigmod(self.wl_linear(left_hidden)+self.wr_linear(right_hidden))
        else:
            bias = self.gl_linear1(left_hidden) + self.gr_linear2(right_hidden)
            bias = F.softmax(bias, dim=1)
            h_tem = self.sigmod(self.wl_linear1(left_hidden) + self.wr_linear2(right_hidden))
        hidden_output = bias[0][0].item()*h_tem + bias[0][1].item()*left_hidden + bias[0][2].item()*right_hidden
        # 第一轮需要用Linear转尺寸
        if is_first:
            hidden_output = self.linearout(hidden_output)
        return hidden_output

# 解码器，采用GRU模型做解码器部分
class Decoder(nn.Module):
    #encoder = Encoder(enb_dim,french_input_size, hidden_dim)
    #decoder = Decoder(enb_dim, chinese_input_size,hidden_dim)
    def __init__(self, enb_dim,  vocab_dim,hidden_dim,output_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_dim, enb_dim)
        self.gru = nn.GRU(enb_dim + hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input,hidden,encoder_out):
        embedded = self.embedding(input)
        num_samples = embedded.size(0)
        encoder_out = encoder_out.repeat(num_samples, 1)
        embedded = torch.cat((embedded,encoder_out),dim=1)
        output, hidden = self.gru(embedded, hidden)
        output = self.linear(output)
        output = self.softmax(output)
        return output,hidden

# 仅用来测试本文件模型是否能正常运行，不是模型训练函数
def train(model, input_seq, target_seq, criterion, optimizer):
    optimizer.zero_grad()
    output = model(input_seq, target_seq)
    loss = criterion(output.view(-1, output.shape[-1]), target_seq.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()



