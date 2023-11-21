# -*- encoding: utf-8 -*-
"""

@File    :   inference.py  
@Modify Time : 2023/11/20 0:58 
@Author  :  Allen.Yang  
@Contact :   MC36514@um.edu.mo        
@Description  : 模型推理文件

"""
import torch
from model.model import Encoder,Decoder,Seq2Seq
from dataloader import preprocess_sentence
from utils import compute_BLEU
import os
import pickle
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
MAX_LENGTH = 50


def find_latest_model():
    model_folder = "./model/model_result/"
    model_files = os.listdir(model_folder)
    if not model_files:
        return None
    model_files.sort(key=lambda x: os.path.splitext(x)[0].split("_")[-1], reverse=True)
    latest_model = None
    max_epoch = float('-inf')
    max_timestamp = os.path.splitext(model_files[0])[0].split("_")[-1]
    #print(f"max_timestamp   {max_timestamp}")

    # 遍历最新的模型文件，找到具有最大epoch的模型
    for model_file in model_files:
        filename, extension = os.path.splitext(model_file)
        epoch = int(filename.split("_")[1][5:])
        timestamp = filename.split("_")[2]
        #print(timestamp)

        if timestamp == max_timestamp and epoch > max_epoch:
            max_epoch = epoch
            latest_model = model_file
    # 从模型文件名中提取出epoch
    return model_folder + latest_model

def inference_sentence(src_sentence,hidden_size,enb_dim,model_path = find_latest_model()):
    print(model_path)

    folder_name = os.path.basename(model_path)
    vocab_path = "./model/vocab_result/"
    #epoch = int(folder_name.split("_")[1][5:])
    # 构建词表文件名
    src_vocab_file = vocab_path + f"vocab_src.pt"
    tgt_vocab_file = vocab_path + f"vocab_tgt.pt"

    # 加载SRC词表
    with open(src_vocab_file, 'rb') as f:
        src_vocab = pickle.load(f)
    # 加载TGT词表
    with open(tgt_vocab_file, 'rb') as f:
        tgt_vocab = pickle.load(f)
    input_size = len(src_vocab)
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    encoder = Encoder(enb_dim, src_vocab_size, hidden_size)
    decoder = Decoder(enb_dim, tgt_vocab_size, hidden_size)
    model = Seq2Seq(encoder, decoder)
    model.load_state_dict(torch.load(model_path),strict=False)
    #print(f"input_size:{input_size},hidden_size{hidden_size}")
    src_sentence = preprocess_sentence(src_sentence)
    print(f"src_sentence:   {src_sentence}")
    src_unk_index = src_vocab['<unk>']
    model.eval()  # 切换为评估模式，不进行梯度计算
    with torch.no_grad():  # 禁用梯度计算
        src_idx = torch.LongTensor([src_vocab.get(i, src_unk_index) for i in src_sentence])
        # 将输入数据传入编码器
        #encoder_outputs, hidden = model.encoder(src_idx.unsqueeze(1))
        vector_list = [x.unsqueeze(0) for x in src_idx]
        is_first_round = True  # 该标代表是否为第一轮，第一轮经过embedding，后面则不用
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
        # 列表只剩下一个值为最终输出
        encoder_final = vector_list[0]
        #output = output.expand(len(tgt_idx), -1)
        # 初始化解码器的输入
        decoder_input = torch.LongTensor([tgt_vocab['<sos>']])
        decoder_hidden = encoder_final
        #print(f"decoder_hidden.size(){decoder_hidden.size()}")
        output_sequence = []
        for _ in range(MAX_LENGTH):
            decoder_output,decoder_hidden = model.decoder(decoder_input, decoder_hidden)  # 解码器输出和隐藏状态
            #print("ok")
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(1)  # 获取预测的单词索引
            output_token = decoder_input.item()
            output_sequence.append(output_token)


            if output_token == tgt_vocab['<eos>']:  # 如果预测到结束符，则停止解码
                print("翻译中止")
                break

            decoder_input = decoder_input
            decoder_hidden = decoder_hidden
        # 将预测的单词索引添加到翻译结果中
        output_sentence = ' '.join([key for token in output_sequence if token not in [tgt_vocab['<sos>'], tgt_vocab['<eos>']] for key, value in tgt_vocab.items() if value == token])
    print(output_sentence)
    return output_sentence


def inference_file(file_path, hidden_size, enb_dim, model_path=find_latest_model()):
    print(model_path)

    folder_name = os.path.basename(model_path)
    vocab_path = "./model/vocab_result/"
    #epoch = int(folder_name.split("_")[1][5:])
    # 构建词表文件名
    src_vocab_file = vocab_path + f"vocab_src.pt"
    tgt_vocab_file = vocab_path + f"vocab_tgt.pt"

    # 加载SRC词表
    with open(src_vocab_file, 'rb') as f:
        src_vocab = pickle.load(f)
    # 加载TGT词表
    with open(tgt_vocab_file, 'rb') as f:
        tgt_vocab = pickle.load(f)
    input_size = len(src_vocab)
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    encoder = Encoder(enb_dim, src_vocab_size, hidden_size)
    decoder = Decoder(enb_dim, tgt_vocab_size, hidden_size)
    model = Seq2Seq(encoder, decoder)
    model.load_state_dict(torch.load(model_path), strict=False)
    #print(f"input_size:{input_size},hidden_size{hidden_size}")
    model.eval()  # 切换为评估模式，不进行梯度计算

    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()

    timestamp = int(time.time())
    output_file_path = f"./translate_result/output_{timestamp}.txt"

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for src_sentence in sentences:
            src_sentence = preprocess_sentence(src_sentence)
            print(f"src_sentence: {src_sentence}")
            src_unk_index = src_vocab['<unk>']
            with torch.no_grad():  # 禁用梯度计算
                src_idx = torch.LongTensor([src_vocab.get(i, src_unk_index) for i in src_sentence])
                # 将输入数据传入编码器
                vector_list = [x.unsqueeze(0) for x in src_idx]
                is_first_round = True  # 该标代表是否为第一轮，第一轮经过embedding，后面则不用
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
                        output_hidden = encoder(embedded_left, embedded_right, is_first_round)
                        count += 1
                        list.append(output_hidden)
                    is_first_round = False
                    vector_list = list
                    step += 1
                # 列表只剩下一个值为最终输出
                encoder_final = vector_list[0]
                # 初始化解码器的输入
                decoder_input = torch.LongTensor([tgt_vocab['<sos>']])
                decoder_hidden = encoder_final
                output_sequence = []
                for _ in range(MAX_LENGTH):
                    decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)  # 解码器输出和隐藏状态
                    _, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze(1)  # 获取预测的单词索引
                    output_token = decoder_input.item()
                    output_sequence.append(output_token)

                    if output_token == tgt_vocab['<eos>']:  # 如果预测到结束符，则停止解码
                        print("翻译中止")
                        break

                    decoder_input = decoder_input
                    decoder_hidden = decoder_hidden
                # 将预测的单词索引添加到翻译结果中
                output_sentence = ' '.join([key for token in output_sequence if token not in [tgt_vocab['<sos>'], tgt_vocab['<eos>']] for key, value in tgt_vocab.items() if value == token])
                f.write(output_sentence + '\n')
    bleu_score = compute_BLEU(file_path,output_file_path)
    print(f"推理结果已写入文件：{output_file_path}")
    print(f"BLEU SCORE:         {bleu_score}")



if __name__ == '__main__':
    #print(find_latest_model())
    sentence = "Je suis étudiant"

    inference_sentence(sentence,128,30)
    inference_file("./datasets/test/test.src.fr",128,30)
