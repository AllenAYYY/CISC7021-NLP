# -*- encoding: utf-8 -*-
"""

@File    :   dataloader.py  
@Modify Time : 2023/11/19 14:34 
@Author  :  Allen.Yang  
@Contact :   MC36514@um.edu.mo        
@Description  : 从数据集中获取数据,协助构建词表和预处理

"""
import random

# import package
import torch
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
from utils import Field
import spacy
import re

# 从spacy加载法语分词器
nlp_fr = spacy.load("fr_core_news_sm")

# 定义法语分词函数
def fr_word_tokenize(text):
    doc = nlp_fr(text)
    return [token.text for token in doc]

# 构建词表的dataloader
class DataLoaderBuildVocab:
    def __init__(self, src_file_path, tgt_file_path, batch_size=2, shuffle=False):
        self.src_file_path = src_file_path
        self.tgt_file_path = tgt_file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.SRC = Field(tokenize_fn=fr_word_tokenize, lower=True, init_token="<sos>", eos_token="<eos>")
        self.TGT = Field(tokenize_fn=word_tokenize, lower=True, init_token="<sos>", eos_token="<eos>")
        self.SRC_VOCAB = None
        self.TGT_VOCAB = None

    # 源语言、目标语言成对输入
    def read_data_to_dataloader(self):
        src_dataset = MyDataset(self.src_file_path)
        tgt_dataset = MyDataset(self.tgt_file_path)
        combined_dataset = list(zip(src_dataset, tgt_dataset))  # 将源数据和目标数据合并
        if self.shuffle:
            random.shuffle(combined_dataset)  # 如果需要打乱顺序，可以在这里进行打乱
        src_dataloader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=False)
        tgt_dataloader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=False)
        return src_dataloader, tgt_dataloader


    def read_to_build_vocab(self):
        src_data = self._read_file(self.src_file_path)
        tgt_data = self._read_file(self.tgt_file_path)
        self.SRC_VOCAB = self.SRC.build_vocab(src_data)
        self.TGT_VOCAB = self.TGT.build_vocab(tgt_data)

    def _read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
        data = [line.strip() for line in data]
        return data

# 预处理，包括替换引号，删掉常用的一些标点符号
# 这一步是给输入的句子做预处理而不是给构建词表做预处理
def preprocess_sentence(sentence,init_token='<sos>',end_token='<eos>'):
    sentence = sentence.replace("'", "")  # 替换单引号为无
    sentence = re.sub(r"([^\s\w]|_)+", " ", sentence)
    # 使用nltk分词工具进行分词
    tokens = word_tokenize(sentence)
    tokens = [token.lower() for token in tokens]
    preprocess_tokens = []
    # 为输入的句子做添加开始符号和结束符号
    if not tokens[0] == init_token:
        preprocess_tokens.append(init_token)
    preprocess_tokens.extend(tokens)
    if not tokens[-1] == end_token:
        preprocess_tokens.append(end_token)
    preprocessed_sentence = ' '.join(preprocess_tokens)
    print(preprocess_tokens)
    return preprocessed_sentence


class MyDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            self.data = file.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index].lower().strip()

def read_data_to_dataloader(file_path,batch_size=2, shuffle=False):
    dataset = MyDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == '__main__':
    src_file_path = "./datasets/train/train.src.fr"
    tgt_file_path = "./datasets/train/train.tgt.en"
    data_loader = DataLoaderBuildVocab(src_file_path, tgt_file_path)
    data_loader.read_to_build_vocab()
    #src_dataloader, tgt_dataloader = data_loader.read_data_to_dataloader()
    for loader in read_data_to_dataloader(src_file_path,3):
        print(loader)
    print(data_loader.SRC_VOCAB)
    print(data_loader.TGT_VOCAB)
    sentence = "Je suis étudiant"
    print(f"处理好的句子{preprocess_sentence(sentence)}")