# -*- encoding: utf-8 -*-
"""

@File    :   model_demo.py  
@Modify Time : 2023/11/6 21:17 
@Author  :  Allen.Yang  
@Contact :   MC36514@um.edu.mo        
@Description  : 部分模型或模型单元的示例代码。

"""
#采用pytorch框架
import torch
import torch.nn as nn


# 定义GRU-based RNN模型
# 注：GRU为论文作者前一篇论文所提出的内容，是LSTM的简化版本
# 注：由重置门和更新门两个门组成，可以选择性的保留或舍弃某些部分(短期记忆)，也可以长期保留某些部分(长期记忆)
# 注：RNN为循环神经网络，整体思想即将之前的内容传输到下一层，达到各模块互相关联影响的效果，和NLP的上下文异曲同工
# 注：论文作者采用结构递归的方式，而常规采用内容递归的方式，二者有区别。
class GRURNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        '''
        :description: 初始化GRU层
        :param input_size: 输入特征的大小
        :param hidden_size: 隐藏状态的大小(隐藏层)
        :param output_size: 输出的大小
        '''
        super(GRURNN, self).__init__()
        # 关于hidden，每个时间步中，GRU层会根据当前输入和上一个时间步的隐藏状态
        # 计算新的隐藏状态，即图中的S。
        self.hidden_size = hidden_size
        # batch_first=True启用分批次，启用时输入数据维度中第一个维度是批次大小
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        # 线性曾self.fc用来将GRU层的输出映射到输出大小为output_size的空间
        self.fc = nn.Linear(hidden_size, output_size)

    # 前向传播
    def forward(self, input):
        batch_size = input.size(0)
        hidden = self.init_hidden(batch_size)
        # input和hidden传入到GRU层,得到输出和最终的隐藏状态
        output, _ = self.gru(input, hidden)
        # 映射输出
        output = self.fc(output[:, -1, :])
        return output

    def init_hidden(self, batch_size):
        '''
        :description: 初始化hidden
        :param batch_size: 一批数据中的样本数量，例如值为2表示训练、推断时会同时处理两个样本
        :return: 返回一个尺寸为(1,batch_size,hidden_size)
        '''
        return torch.zeros(1, batch_size, self.hidden_size)


class RecursiveNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RecursiveNN, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, children=None):
        # 如果不存在子节点，则返回input
        # 因为叶子节点没有子节点的信息要考虑
        if children is None or len(children) == 0:
            return input

        # 将子节点的表示递归地传入递归神经网络
        # 对每个子节点进行递归调用，子节点的输出存储在child_reps中
        child_reps = [self.forward(child) for child in children]

        # 将子节点的表示进行拼接
        # dim=0按行进行拼接
        # 注：这部分代码尚为我自己的猜测，并不确定真实实现是否如此
        child_concat = torch.cat(child_reps, dim=0)
        child_sum = torch.sum(child_concat, dim=0)

        # 将当前节点的表示和子节点的汇总表示进行拼接
        combined = torch.cat([input, child_sum], dim=0)

        # 通过线性层得到当前节点的表示
        output = self.linear(combined)

        return output
# 示例用法
input_size = 10
hidden_size = 20
output_size = 5
sequence_length = 30
batch_size = 2

# 创建模型实例
model = GRURNN(input_size, hidden_size, output_size)

# 创建随机输入数据
# 输入数据大小为(batch_size,sequence_length,input_size)
# 输入数据为一个3D张量，分别为批次大小、序列长度和输入特征大小
# sequence_length: 序列长度，即时间步的数量。
# input_size表示每个时间步输入的特征维度。
input_data = torch.randn(batch_size, sequence_length, input_size)

# 前向传播计算
output = model(input_data)
print(output.shape)  # 输出: torch.Size([2, 5])，表示两个样本的预测结果