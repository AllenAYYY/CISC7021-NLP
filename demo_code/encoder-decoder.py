import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
from nltk import FreqDist

embedding_dim = 100
hidden_dim = 128
max_tokens = 100  # 最大标记序列长度

class TokenizerWrap(nn.Module):
    def __init__(self, texts, padding, reverse=False, num_words=None):
        super(TokenizerWrap, self).__init__()

        self.tokenizer = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.reverse = reverse
        self.padding = padding

        if num_words is not None:
            self.vocab = self.build_vocab(texts, num_words)
        else:
            self.vocab = self.build_vocab(texts)

    def build_vocab(self, texts, num_words=None):
        words = [word for text in texts for word in word_tokenize(text)]
        fdist = FreqDist(words)

        if num_words is not None:
            vocab = [word for word, _ in fdist.most_common(num_words)]
        else:
            vocab = [word for word, _ in fdist.most_common()]

        return vocab

    def forward(self, tokens):
        embedded = self.tokenizer(tokens)
        return embedded

    def token_to_word(self, token):
        word = self.vocab[token]
        return word

    def tokens_to_string(self, tokens):
        words = [self.vocab[token] for token in tokens if token != 0]
        text = ' '.join(words)
        return text

    def text_to_tokens(self, text):
        tokens = [self.vocab.index(word) for word in word_tokenize(text)]

        if self.reverse:
            tokens = list(reversed(tokens))
            truncating = 'pre'
        else:
            truncating = 'post'

        if truncating == 'pre':
            tokens = [0] * max(0, max_tokens - len(tokens)) + tokens[:max_tokens]
        else:
            tokens = tokens[:max_tokens] + [0] * max(0, max_tokens - len(tokens))

        tokens = torch.tensor(tokens).unsqueeze(0)

        return tokens


# 示例用法
input_text = "This is a test sentence."

# 创建 TokenizerWrap 对象
tokenizer = TokenizerWrap(texts=[input_text],
                          padding='pre',
                          reverse=True,
                          num_words=None)

# 将输入字符串转换为填充后的标记序列
tokens = tokenizer.text_to_tokens(input_text)

# 打印填充后的标记序列及其形状
print(tokens)
print(tokens.shape)

# 将填充后的标记序列转换回字符串
output_text = tokenizer.tokens_to_string(tokens[0])
print(output_text)