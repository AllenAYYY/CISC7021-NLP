# -*- encoding: utf-8 -*-
"""

@File    :   train_demo.py.py
@Modify Time : 2023/11/15 20:29
@Author  :  Allen.Yang
@Contact :   MC36514@um.edu.mo
@Description  : 描述代码训练过程的大概流程

"""
import torch
from torch.utils.data import DataLoader

import torch.optim as optim
from model_demo import GRURNN,RecursiveNN
from config import parse_arguments

def main():
    # 创建命令行参数解析器并解析参数
    args = parse_arguments()

    # 从命令行参数获取设定值
    learning_rate = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    test_interval = args.test_interval
    resume_model_path = args.resume_model

    # 创建模型
    rnn_model = GRURNN
    gru_model = RecursiveNN

    # ...

    # 定义损失函数和优化器
    # 使用从命令行参数获取的学习率
    optimizer_rnn = optim.Adam(rnn_model.parameters(), lr=learning_rate)
    optimizer_gru = optim.Adam(gru_model.parameters(), lr=learning_rate)

    # ...

    # 模型训练
    start_epoch = 1
    if resume_model_path:
        # 如果指定了从中断点继续训练的模型，加载模型参数
        checkpoint = torch.load(resume_model_path)
        rnn_model.load_state_dict(checkpoint['rnn_model_state_dict'])
        gru_model.load_state_dict(checkpoint['gru_model_state_dict'])
        optimizer_rnn.load_state_dict(checkpoint['optimizer_rnn_state_dict'])
        optimizer_gru.load_state_dict(checkpoint['optimizer_gru_state_dict'])
        start_epoch = checkpoint['epoch'] + 1



    for epoch in range(start_epoch, num_epochs + 1):
        rnn_model.train()
        gru_model.train()
        rnn_loss = 0
        gru_loss = 0


        #训练迭代，只用于描述，非具体实现
        train_iterator = []
        for batch in train_iterator:
            continue
            # TO DO ......

        rnn_loss /= len(train_iterator)
        gru_loss /= len(train_iterator)

        print(f"Epoch: {epoch}, RNN Loss: {rnn_loss:.4f}, GRU Loss: {gru_loss:.4f}")

        if epoch % test_interval == 0:
            rnn_model.eval()
            gru_model.eval()
            # 进行测试并保存模型
            # ...

            # 保存模型参数
            checkpoint = {
                'epoch': epoch,
                'rnn_model_state_dict': rnn_model.state_dict(),
                'gru_model_state_dict': gru_model.state_dict(),
                'optimizer_rnn_state_dict': optimizer_rnn.state_dict(),
                'optimizer_gru_state_dict': optimizer_gru.state_dict()
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pt")

if __name__ == '__main__':
    main()