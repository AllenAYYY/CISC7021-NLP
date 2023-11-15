# -*- encoding: utf-8 -*-
"""

@File    :   GRU_demo.py  
@Modify Time : 2023/11/15 17:34 
@Author  :  Allen.Yang  
@Contact :   MC36514@um.edu.mo        
@Description  : GRU非库函数实现，主要包括重置门、更新门、隐藏候选三个部分

"""

import torch
import torch.nn as nn
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        '''
        :description: 初始化
        :param input_size: 输入尺寸，即图中x_t-1
        :param hidden_size: 隐藏层尺寸，即图中h_t-1
        '''
        # 注：t为时间步
        super(GRU, self).__init__()

        self.hidden_size = hidden_size

        # 重置门参数
        # Linear会更新权重矩阵和偏置两个部分，它内部维护这两个模块，对输入内容进行乘权重再加偏置
        # Linear第一次实例化的时候会随机生成参数，第一个参数为输入尺寸，第二个为输出尺寸
        # 重置门步骤即图中的r_t部分
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        # Sigmoid即激活函数，将数据映射为0-1之间
        self.reset_activation = nn.Sigmoid()

        # 更新门参数
        # 即图中z_t部分
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_activation = nn.Sigmoid()

        # 候选隐藏状态参数
        # 即图中h_t部分
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        # 按图实现，传入重置门、更新门都使用了拼接
        # dim=1按列方向拼接，一般都是按列拼接
        combined = torch.cat((input, hidden), dim=1)

        # 计算重置门，即乘权重再+偏置
        reset = self.reset_gate(combined)
        # Sigmod激活
        reset = self.reset_activation(reset)

        # 计算更新门
        update = self.update_gate(combined)
        update = self.update_activation(update)

        # 计算候选隐藏状态
        # 候选隐藏层拼接重置门结果以及输入
        combined = torch.cat((input, reset * hidden), dim=1)
        candidate = self.candidate(combined)
        candidate = self.tanh(candidate)

        # 更新隐藏状态，即h_t
        output = (1 - update) * hidden + update * candidate

        return output

# 示例用法
input_size = 10
hidden_size = 20
batch_size = 32
sequence_length = 5

# 一般randn用来生成0、1正态分布，用于输入数据
input = torch.randn(batch_size,  input_size)
# 一般模型中间层使用0初始化
hidden = torch.zeros(batch_size, hidden_size)

gru = GRU(input_size, hidden_size)
output = gru.forward(input, hidden)
print("Output shape:", output.shape)


