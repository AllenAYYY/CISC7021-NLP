# -*- encoding: utf-8 -*-
"""

@File    :   utils.py  
@Modify Time : 2023/11/15 20:44 
@Author  :  Allen.Yang  
@Contact :   MC36514@um.edu.mo        
@Description  : 此文件提供工具函数，比如从文件读取数据、构建词表、大小写转化等

"""
# import package
import nltk
from nltk.tokenize import word_tokenize

# 数据预处理
class Field:
    def __init__(self, tokenize_fn=None, lower=False, init_token=None, eos_token=None):
        self.tokenize_fn = tokenize_fn
        self.lower = lower
        self.init_token = init_token
        self.eos_token = eos_token
        self.vocab = {}

    # 构建词表
    def build_vocab(self, data):
        tokens = []
        # 添加开始符号和结束符号
        if self.init_token:
            tokens.append(self.init_token)
        if self.eos_token:
            tokens.append(self.eos_token)
        # 对句子做分词
        for sentence in data:
            sentence_tokens = self.tokenize_fn(sentence)
            # 转小写
            if self.lower:
                sentence_tokens = [token.lower() for token in sentence_tokens]
            tokens.extend(sentence_tokens)
        # 构建字典
        freq_dist = nltk.FreqDist(tokens)
        # 生成词表
        self.vocab = {token: idx for idx, (token, _) in enumerate(freq_dist.items())}
        # 添加未知词符号
        self.vocab['<unk>'] = len(self.vocab)
        return self.vocab

    def numericalize(self, data):
        if self.lower:
            data = [token.lower() for token in data]
        if self.init_token:
            data = [self.init_token] + data
        if self.eos_token:
            data = data + [self.eos_token]
        return [self.vocab[token] for token in data]

if __name__ == '__main__':
    # 定义数据处理字段
    SRC = Field(tokenize_fn=word_tokenize, lower=True, init_token="<sos>", eos_token="<eos>")
    TGT = Field(tokenize_fn=word_tokenize, lower=True, init_token="<sos>", eos_token="<eos>")

    # 加载并处理数据集
    train_src_data = [
        "Je suis étudiant.",
        "Elle aime danser.",
        "Nous avons mangé une pizza.",
        "Il fait beau aujourd'hui.",
    ]

    train_tgt_data = [
        "I am a student.",
        "She loves to dance.",
        "We ate a pizza.",
        "The weather is nice today.",
    ]

    valid_src_data = [
        "Je suis heureux.",
        "Elle parle français.",
    ]

    valid_tgt_data = [
        "I am happy.",
        "She speaks French.",
    ]

    test_src_data = [
        "Je suis fatigué.",
        "Elle aime lire.",
    ]

    # 创建数据处理字段的词汇表
    SRC.build_vocab(train_src_data)
    TGT.build_vocab(train_tgt_data)

    # 打印词汇表
    print(SRC.vocab)
    print(TGT.vocab)
