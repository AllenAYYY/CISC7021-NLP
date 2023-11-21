# -*- encoding: utf-8 -*-
"""

@File    :   utils.py  
@Modify Time : 2023/11/15 20:44 
@Author  :  Allen.Yang  
@Contact :   MC36514@um.edu.mo        
@Description  : 此文件提供工具函数，比如从文件读取数据、构建词表、大小写转化等

"""
# import package
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
from model.model import Encoder,Decoder,Seq2Seq
from dataloader import DataLoaderBuildVocab,read_data_to_dataloader
from nltk.translate.bleu_score import corpus_bleu



# 数据预处理


# 测试模型
def test_model(src_path, tgt_path, batch_size,enb_dim,hidden_dim,mod):
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
                # 列表只剩下一个值为最终输出
                output = vector_list[0]
                # print(f"outputsize{output.size()}")
                output, hidden = decoder(tgt_idx, output)

                # 计算模型输出的序列和目标语言的序列之间的loss
                loss = criterion(output, tgt_idx)
                #loss = criterion(output.view(-1, output.shape[-1]), tgt_idx.view(-1))
                total_loss += loss.item() * len(src_idx)
                total_samples += len(src_idx)
    # 计算平均损失
    avg_loss = total_loss / total_samples
    if mod == 'test':
        print(f"test Loss: {avg_loss:.4f}")
    else:
        print(f"valid Loss: {avg_loss:.4f}")


def compute_BLEU(src_filepath,tgt_filepath):
    with open(src_filepath, 'r', encoding='utf-8') as ref_file:
        reference_data = ref_file.readlines()

    with open(tgt_filepath, 'r', encoding='utf-8') as cand_file:
        candidate_data = cand_file.readlines()

    # 将参考和候选数据转换为适合计算 BLEU 的格式
    references = [[ref.strip().split()] for ref in reference_data]
    candidates = [cand.strip().split() for cand in candidate_data]

    # 计算 BLEU 分数
    bleu_score = corpus_bleu(references, candidates)

    return bleu_score







