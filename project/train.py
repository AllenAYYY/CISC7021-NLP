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
import pickle
from model.model import Encoder,Decoder,Seq2Seq
from dataloader import DataLoaderBuildVocab,read_data_to_dataloader,preprocess_sentence
import os

# 环境变量防止.ddl重复
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 翻译限制最大句子
MAX_LENGTH = 50

# 文件路径
src_file_path = "./datasets/train/train.src.fr"
tgt_file_path = "./datasets/train/train.tgt.en"

# 初始化DataLoader
data_loader = DataLoaderBuildVocab(src_file_path, tgt_file_path)
data_loader.read_to_build_vocab()

# 源语言词表 & 目标语言词表
SRC_VOCAB = data_loader.SRC_VOCAB
TGT_VOCAB = data_loader.TGT_VOCAB

# 测试模型
def test_model(src_path, tgt_path, batch_size,enb_dim,hidden_dim):
    '''
    :description: 用以实现模型测试
    :param src_path: 验证集源语言的src_path
    :param tgt_path:  验证集目标语言的tgt_path
    :param batch_size: 批处理大小
    :param enb_dim: 词映射为vector的尺寸
    :param hidden_dim: 隐藏层维度
    :return: None. Only print
    '''
    data_loader = DataLoaderBuildVocab(src_path, tgt_path)
    data_loader.read_to_build_vocab()
    src_vocab = data_loader.SRC_VOCAB
    tgt_vocab = data_loader.TGT_VOCAB
    src_vocab_size = len(data_loader.SRC_VOCAB)
    tgt_vocab_size = len(data_loader.TGT_VOCAB)
    # 实例化 编码器和解码器 和模型
    encoder = Encoder(enb_dim, src_vocab_size, hidden_dim)
    decoder = Decoder(enb_dim, tgt_vocab_size, hidden_dim)
    model = Seq2Seq(encoder, decoder)
    model.eval()  # 切换为评估模式，不进行梯度计算
    criterion = nn.NLLLoss() # 采用nn.NLLLOSS()计算损失
    total_loss = 0
    total_samples = 0

    with torch.no_grad():  # 禁用梯度计算
        for loader_src, loader_tgt in zip(read_data_to_dataloader(src_path, batch_size),read_data_to_dataloader(tgt_path,batch_size)):
            for src_item, tgt_item in zip(loader_src, loader_tgt):
                # 分词后的句子，根据对应语言的词表，生成代表词含义的vector
                src_idx = torch.LongTensor([src_vocab[i] for i in src_item.split()])
                tgt_idx = torch.LongTensor([tgt_vocab[i] for i in tgt_item.split()])
                # 将每一个节点的vector传入list中。
                vector_list = [x.unsqueeze(0) for x in src_idx]
                # 第一轮要经过embedding层，后续则不需要
                is_first_round = True
                # 当列表中只有一个节点时代表递归完成
                while len(vector_list) > 1:
                    num_vector = len(vector_list)
                    count = 0
                    step = 0
                    list = []
                    # 判断存在下一个节点
                    while count + 1 < num_vector:
                        if is_first_round: # 第一轮要经过embedding层，后续则不需要
                            embedded_left = encoder.embeddingl(vector_list[count])
                            embedded_right = encoder.embeddingr(vector_list[count + 1])
                        else:
                            embedded_left = vector_list[count]
                            embedded_right = vector_list[count + 1]
                        # 传入模型
                        output_hidden = model(embedded_left, embedded_right, tgt_idx, is_first_round)
                        #print(f"count{count}成功")
                        count += 1
                        #print(f"output_hidden.size()    {output_hidden.size()}")
                        list.append(output_hidden)
                    is_first_round = False
                    vector_list = list
                    step += 1
                # 列表只剩下一个值为最终输出
                output = vector_list[0]
                output = output.expand(len(tgt_idx), -1)
                # 计算模型输出的序列和目标语言的序列之间的loss
                loss = criterion(output, tgt_idx)
                #loss = criterion(output.view(-1, output.shape[-1]), tgt_idx.view(-1))
                total_loss += loss.item() * len(src_idx)
                total_samples += len(src_idx)
    # 计算平均损失
    avg_loss = total_loss / total_samples
    print(f"valid test Loss: {avg_loss:.4f}")

# 保存词表
def save_vocab(vocab,epoch,tag):
    file_path = "./model/vocab_result/" + f"vocab_epoch{epoch}_{tag}.pt"
    with open(file_path, 'wb') as f:
        pickle.dump(vocab, f)

# 训练模型
def train_model(train_src_path,train_tgt_path,valid_src_path,valid_tgt_path,learning_rate = 0.001,batch_size=2,num_epoch=70,enb_dim = 30,hidden_dim=128):
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
    src_vocab = SRC_VOCAB
    tgt_vocab = TGT_VOCAB
    save_vocab(SRC_VOCAB,num_epoch,"src")
    save_vocab(TGT_VOCAB,num_epoch,"tgt")
    print(src_vocab)
    src_vocab_size = len(data_loader.SRC_VOCAB)
    tgt_vocab_size = len(data_loader.TGT_VOCAB)
    # 实例化编码器、解码器和模型
    encoder = Encoder(enb_dim, src_vocab_size, hidden_dim)
    decoder = Decoder(enb_dim, tgt_vocab_size, hidden_dim)
    model = Seq2Seq(encoder, decoder)
    # Adam优化
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss() # 采用NLLLOSS损失函数
    loss_result = 0
    loss_history = []
    for epoch in range(1,num_epoch+1):
        # 加载批次loader
        for loader_src,loader_tgt in zip(read_data_to_dataloader(train_src_path, batch_size),read_data_to_dataloader(train_tgt_path,batch_size)):
            #print(f"源{loader_src},目标{loader_tgt}")
            # 加载批中的每一对训练语言
            for src_item,tgt_item in zip(loader_src,loader_tgt):
                #print(f"源{src_item},目标{tgt_item}")
                # 从词表获取语言序列
                src_idx = torch.LongTensor([src_vocab[i] for i in src_item.split()])
                tgt_idx = torch.LongTensor([tgt_vocab[i] for i in tgt_item.split()])
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
                        if is_first_round: # 第一轮经过embedding将词序列编码
                            embedded_left = encoder.embeddingl(vector_list[count])
                            embedded_right = encoder.embeddingr(vector_list[count + 1])
                        else: # only第一轮需要，后面不需要
                            embedded_left = vector_list[count]
                            embedded_right = vector_list[count + 1]
                        output_hidden = model(embedded_left, embedded_right, tgt_idx, is_first_round)
                        count += 1
                        list.append(output_hidden)
                    is_first_round = False
                    vector_list = list
                    step += 1
                output = vector_list[0]
                output = output.expand(len(tgt_idx), -1)

                # 解码器输出和目标语言的序列计算loss
                loss = criterion(output, tgt_idx)
                loss.backward()
                loss_result = loss.item()
                print(f"第{epoch}轮 Train Loss: {loss_result}")
                optimizer.step()
        loss_history.append(loss_result)
        if epoch  % 10 == 0:
            print(f"Epoch: {epoch}, Valid Loss: {loss_result:.4f}")
            test_model(valid_src_path,valid_tgt_path,batch_size,enb_dim,hidden_dim)
        if epoch % 50 == 0:
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
    train_model(src_file_path, tgt_file_path, valid_src_path, valid_tgt_path, batch_size=50)

