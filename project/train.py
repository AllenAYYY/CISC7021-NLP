# -*- encoding: utf-8 -*-
"""

@File    :   train.py  
@Modify Time : 2023/11/15 20:45
@Author  :  Allen.Yang
@Contact :   MC36514@um.edu.mo
@Description  : 此文件用于实现模型训练、测试

"""
# import package
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime
import time
from model.model import Encoder,Decoder,Seq2Seq
from dataloader import DataLoaderBuildVocab,read_data_to_dataloader,preprocess_sentence,read_vocab,save_vocab
from utils import test_model
import os

# 环境变量防止.ddl重复
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 翻译限制最大句子
MAX_LENGTH = 50

# 文件路径
src_file_path = "./datasets/train/train.src.fr"
tgt_file_path = "./datasets/train/train.tgt.en"


src_vocab_file = "./model/vocab_result/vocab_src.pt"
tgt_vocab_file = "./model/vocab_result/vocab_tgt.pt"



# 训练模型
def train_model(train_src_path,train_tgt_path,valid_src_path,valid_tgt_path,learning_rate = 0.001,batch_size=64,num_epoch=70,enb_dim = 10,hidden_dim=16):
    '''
    :description: 实现模型的训练过程
    :param train_src_path: 训练集src_path
    :param train_tgt_path: 测试集tgt_path
    :param valid_src_path: 验证集src_path
    :param valid_tgt_path: 验证集tgt_path
    :param learning_rate: 学习率
    :param batch_size: 批处理大小
    :param num_epoch: 训练轮数
    :param enb_dim: 词映射到vector的维度
    :param hidden_dim: 隐藏层维度
    :return: None. Only print
    '''

    # 检查源语言词表和目标语言词表文件是否同时存在
    if os.path.exists(src_vocab_file) and os.path.exists(tgt_vocab_file):
        src_vocab = read_vocab(src_vocab_file)
        tgt_vocab = read_vocab(tgt_vocab_file)
        print("词表存在，不需要重新生成")
        #print(src_vocab)
        #print(tgt_vocab)
    else:
        # 初始化DataLoader
        data_loader = DataLoaderBuildVocab(src_file_path, tgt_file_path)
        data_loader.read_to_build_vocab()
        src_vocab = data_loader.SRC_VOCAB
        tgt_vocab = data_loader.TGT_VOCAB
        save_vocab(src_vocab, "src","train")
        save_vocab(tgt_vocab, "tgt","train")
        print("词表不存在，需要重新生成")
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    print(src_vocab)
    print(tgt_vocab)
    # 实例化编码器、解码器和模型
    encoder = Encoder(enb_dim, src_vocab_size, hidden_dim)
    decoder = Decoder(enb_dim, tgt_vocab_size, hidden_dim,tgt_vocab_size)
    model = Seq2Seq(encoder, decoder)
    # Adam优化
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss() # 采用NLLLOSS损失函数
    loss_result = 0
    loss_history = []
    src_unk_index = src_vocab['<unk>']
    tgt_unk_index = tgt_vocab['<unk>']
    for epoch in range(1,num_epoch+1):
        start_time = time.time()  # 记录开始时间
        # 加载批次loader
        for loader_src,loader_tgt in zip(read_data_to_dataloader(train_src_path, batch_size),read_data_to_dataloader(train_tgt_path,batch_size)):
            #print(f"源{loader_src},目标{loader_tgt}")
            # 加载批中的每一对训练语言
            loader_count = 0
            for src_item,tgt_item in zip(loader_src,loader_tgt):
                src_item = preprocess_sentence(src_item)
                tgt_item = preprocess_sentence(tgt_item)
                if src_item == None or tgt_item == None:
                    continue
                #print(f"源{src_item},目标{tgt_item}")
                # 从词表获取语言序列
                # 找不到就转成未知词
                src_idx = torch.LongTensor([src_vocab.get(i, src_unk_index) for i in src_item.split()])
                #print(src_idx)
                tgt_idx = torch.LongTensor([tgt_vocab.get(i, tgt_unk_index) for i in tgt_item.split()])
                optimizer.zero_grad()
                # 传入列表
                # 改处while循环和model中的编码器部分共同构成了论文中的递归神经网络结构
                # 总体思想为：同层每两个节点之间进行运算，结果输入至上一层成为上一层的输入
                # 迭代 until 只剩下一个节点，此时节点的输出代表编码器的输出
                vector_list = [x.unsqueeze(0) for x in src_idx]
                is_first_round = True # 该标代表是否为第一轮，第一轮经过embedding，后面则不用
                # 只剩一个节点时完成迭代
                while len(vector_list) > 1:
                    num_vector = len(vector_list)
                    count = 0
                    step = 0
                    list = []
                    # 不存在下一个节点时迭代完成
                    while count + 1 < num_vector:
                        if is_first_round:  # 第一轮要经过embedding层，后续则不需要
                            embedded_left = encoder.embeddingl(vector_list[count])
                            embedded_right = encoder.embeddingr(vector_list[count + 1])
                        else:
                            embedded_left = vector_list[count]
                            embedded_right = vector_list[count + 1]
                        # 传入模型
                        # output_hidden = model(embedded_left, embedded_right, tgt_idx, is_first_round)
                        output_hidden = encoder(embedded_left, embedded_right, is_first_round)
                        # print(f"count{count}成功")
                        count += 1
                        # print(f"output_hidden.size()    {output_hidden.size()}")
                        list.append(output_hidden)
                    is_first_round = False
                    vector_list = list
                    step += 1
                if vector_list == None or len(vector_list)==0:
                    continue
                # 列表只剩下一个值为最终输出
                output = vector_list[0]
                encoder_output = output
                output, hidden = decoder(tgt_idx, output,encoder_output)
                # 解码器输出和目标语言的序列计算loss
                loss = criterion(output, tgt_idx)
                loss.backward()
                loss_result = loss.item()
                optimizer.step()
            print(f"第{loader_count+1}个loader完成")
            loader_count += 1
        loss_history.append(loss_result)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时
        minutes = int(elapsed_time // 60)  # 计算分钟数
        seconds = int(elapsed_time % 60)  # 计算剩余的秒数

        print(f"第{epoch}轮 Train Loss: {loss_result}    耗时: {minutes}分{seconds}秒")
        if epoch  % 100 == 0:
            #print(f"Epoch: {epoch}, Valid Loss: {loss_result:.4f}")
            test_model(valid_src_path,valid_tgt_path,batch_size,enb_dim,hidden_dim,"valid")
        if epoch % 100 == 0:
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            model_path = f"./model/model_result/model_epoch{epoch}_{current_time}.pt"
            torch.save(model.state_dict(), model_path)

    # 绘制损失函数曲线
    plt.plot(range(1, num_epoch + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("./result_imgs/epoch_" + str(num_epoch) + "_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".png")



if __name__ == '__main__':
    src_file_path = "./datasets/train/train.src.fr"
    tgt_file_path = "./datasets/train/train.tgt.en"
    valid_src_path = "./datasets/valid/valid.src.fr"
    valid_tgt_path = "./datasets/valid/valid.tgt.en"
    test_src_path = "./datasets/test/test.src.fr"
    test_tgt_path = "./datasets/test/test.tgt.en"
    train_model(src_file_path, tgt_file_path, valid_src_path, valid_tgt_path,num_epoch=10000)


