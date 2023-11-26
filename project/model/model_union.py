# -*- encoding: utf-8 -*-
"""

@File    :   model_union.py  
@Modify Time : 2023/11/19 16:39 
@Author  :  Allen.Yang  
@Contact :   MC36514@um.edu.mo        
@Description  : 模型构成模块

"""
# import package
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

class RecursiveNN(nn.Module):
    def __init__(self, input_size, output_size, num_linear,num_relu):
        super(RecursiveNN, self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(num_linear)])
        self.relu_layers = nn.ModuleList([nn.ReLU() for _ in range(num_relu)])

    def forward(self, inputs):
        # 列表里面不止一个张量
        linear_count = 0
        relu_count = 0
        inputs_count = 0
        list_vector = []
        while isinstance(inputs, list) and len(inputs)>1:
            #list_vecotr = []
            # 递归处理输入的左子树和右子树
            left = self.linear_layers[linear_count](inputs[inputs_count])
            #left = self.relu_layers[linear_count](left)
            linear_count += 1
            right = self.linear_layers[linear_count](inputs[inputs_count+1])
            #right = self.relu_layers[linear_count](right)
            linear_count += 1
            inputs_count += 1
            combined = left+right
            combined = self.relu_layers[relu_count](combined)
            relu_count += 1
            list_vector.append(combined)
            if(len(inputs) == inputs_count+1):
                inputs = list_vector
                list_vector = []
                inputs_count  = 0
        # 计算最终的输出
        output = inputs[0]
        return output

def compute_linear_relu_num(input_data):
    len_vecotr = len(input_data)
    if (len_vecotr == 1):
        linear_num = 1
        relu_num = 1
    else:
        end = (len_vecotr - 2) * 2 + 2
        step = 2
        total = 0
        current = 2

        while current <= end:
            total += current
            current += step

        linear_num = total
        print(len_vecotr - 1)
        relu_num = int((len_vecotr - 1 + 1) * (len_vecotr - 1) / 2)
        return linear_num,relu_num

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)

    def forward(self, input, hidden):
        output = []
        for i in range(input.size(0)):
            hidden = self.rnn_cell(input[i], hidden)
            output.append(hidden)
        output = torch.stack(output)
        return output, hidden