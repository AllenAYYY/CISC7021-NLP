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
        output = self.decoder(tgt,encoder_output,encoder_output)
        return output

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
        #self.softmax = nn.LogSoftmax(dim=1)
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
        #print(f"bias.size()     {bias.size()}")
        #print(f"bias     {bias}")
        #print(f"left_hidden.size()     {left_hidden.size()}")
        #print(f"right_hidden.size()     {right_hidden.size()}")
        #print(f"h_tem.size()                  {h_tem.size()}")
        #print(bias[0][1].item())
        #print(h_tem.size())
        hidden_output = bias[0][1].item()*h_tem + bias[0][1].item()*left_hidden + bias[0][2].item()*right_hidden
        #print(f"hidden_output****        {hidden_output}")
        # 第一轮需要用Linear转尺寸
        if is_first:
            hidden_output = self.linearout(hidden_output)
        return hidden_output

# 解码器，采用GRU模型做解码器部分
class Decoder(nn.Module):
    #encoder = Encoder(enb_dim,french_input_size, hidden_dim)
    #decoder = Decoder(enb_dim, chinese_input_size,hidden_dim)
    def __init__(self, enb_dim,  vocab_dim,hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_dim, enb_dim)
        self.gru = nn.GRU(enb_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, tgt, hidden, encoder_output):
        embedded = self.embedding(tgt)
        output, hidden = self.gru(embedded, hidden)
        prediction = self.linear(hidden)
        prediction = self.softmax(prediction)
        return prediction

# 仅用来测试本文件模型是否能正常运行，不是模型训练函数
def train(model, input_seq, target_seq, criterion, optimizer):
    optimizer.zero_grad()
    output = model(input_seq, target_seq)
    loss = criterion(output.view(-1, output.shape[-1]), target_seq.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()



if __name__ == '__main__':
    hidden_dim = 256  # 隐藏层的维度
    learning_rate = 0.001
    enb_dim =30
    epochs = 10
    vocab = {'I': 0, 'am': 1, 'a': 2, 'student': 3, '.': 4, 'She': 5, 'loves': 6, 'to': 7, 'dance': 8, 'We': 9,
             'ate': 10, 'pizza': 11, 'The': 12, 'weather': 13, 'is': 14, 'nice': 15, 'today': 16, '<PAD>': 17}
    french_size_decoder = len(vocab)
    text = 'I am a student <PAD> <PAD>'
    hello_idx = torch.LongTensor([vocab[i] for i in text.split()])
    chinese_vocab = {'我': 0, '是': 1, '一个': 2, '学生': 3, '。': 4, '她': 5, '喜欢': 6, '跳舞': 7, '我们': 8, '吃': 9,
                     '比萨': 10, '天气': 11, '很': 12, '好': 13, '今天': 14, '<PAD>': 15}
    chinese_input_size = len(chinese_vocab)
    french_input_size = len(vocab)
    encoder = Encoder(enb_dim,french_input_size, hidden_dim)
    decoder = Decoder(enb_dim, chinese_input_size,hidden_dim)
    #print(french_input_size,hidden_dim,chinese_input_size)
    model = Seq2Seq(encoder, decoder)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    # 定义英文句子
    english_text = 'I am a student .'
    english_idx = torch.LongTensor([vocab[i] for i in english_text.split()])
    print(english_idx)

    chinese_text = '我 是 学生'
    chinese_idx = torch.LongTensor([chinese_vocab[i] for i in chinese_text.split()])


    vector_list = [x.unsqueeze(0) for x in english_idx]
    #print(vector_list)
    is_first_round = True  # 标记是否是第一轮循环
    while len(vector_list) > 1:
        num_vector = len(vector_list)
        count = 0
        step = 0
        list = []
        while count + 1 <num_vector:
            if is_first_round:
                embedded_left = encoder.embeddingl(vector_list[count])
                embedded_right = encoder.embeddingr(vector_list[count + 1])
            else:
                embedded_left = vector_list[count]
                embedded_right = vector_list[count + 1]

            output_hidden = model(embedded_left, embedded_right, chinese_idx,is_first_round)
            #print(f"count{count}成功")
            count += 1
            #print(f"output_hidden.size()    {output_hidden.size()}")
            list.append(output_hidden)
        is_first_round = False
        vector_list = list
        step += 1
        #print(f"{step}成功！")

    print(vector_list[0])
