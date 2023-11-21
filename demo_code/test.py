import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        output, hidden = self.gru(embedded)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output))
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        enc_output, enc_hidden = self.encoder(input_seq)
        dec_output, dec_hidden = self.decoder(target_seq, enc_hidden)
        return dec_output

def train(model, input_seq, target_seq, criterion, optimizer):
    optimizer.zero_grad()
    output = model(input_seq, target_seq)
    loss = criterion(output.view(-1, output.shape[-1]), target_seq.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()
"""
def train_model(train_src_path,train_tgt_path,valid_src_path,valid_tgt_path,learning_rate = 0.001,batch_size=2,num_epoch=70,hidden_dim=30):
    src_vocab = SRC_VOCAB
    tgt_vocab = TGT_VOCAB
    save_vocab(SRC_VOCAB,num_epoch,"src")
    save_vocab(TGT_VOCAB,num_epoch,"tgt")
    print(src_vocab)
    HIDDEN_SIZE = hidden_dim
    src_vocab_size = len(data_loader.SRC_VOCAB)
    tgt_vocab_size = len(data_loader.TGT_VOCAB)
    encoder = Encoder(src_vocab_size, hidden_dim)
    decoder = Decoder(hidden_dim, tgt_vocab_size)
    model = Seq2Seq(encoder, decoder)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    loss_result = 0
    loss_history = []
    for epoch in range(1,num_epoch+1):
        #print(f"第{epoch}轮")
        for loader_src,loader_tgt in zip(read_data_to_dataloader(train_src_path, batch_size),read_data_to_dataloader(train_tgt_path,batch_size)):
            #print(loader_src,loader_tgt)
            #print(f"源{loader_src},目标{loader_tgt}")
            for src_item,tgt_item in zip(loader_src,loader_tgt):
                #print(src_item,tgt_item)
                print(f"源{src_item},目标{tgt_item}")
                src_idx = torch.LongTensor([src_vocab[i] for i in src_item.split()])
                tgt_idx = torch.LongTensor([tgt_vocab[i] for i in tgt_item.split()])
                optimizer.zero_grad()
                output = model(src_idx, tgt_idx)


                loss = criterion(output.view(-1, output.shape[-1]), tgt_idx.view(-1))
                loss.backward()
                loss_result = loss.item()
                optimizer.step()
        loss_history.append(loss_result)

        if epoch  % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss_result:.4f}")
            test_model(valid_src_path,valid_tgt_path)
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



# 定义测试函数
def test_encoder():
    # 设置输入参数
    hidden_dim = 256  # 隐藏层的维度
    learning_rate = 0.001
    epochs = 10
    vocab = {'I': 0, 'am': 1, 'a': 2, 'student': 3, '.': 4, 'She': 5, 'loves': 6, 'to': 7, 'dance': 8, 'We': 9, 'ate': 10, 'pizza': 11, 'The': 12, 'weather': 13, 'is': 14, 'nice': 15, 'today': 16, '<PAD>': 17}
    french_size_decoder = len(vocab)
    text = 'I am a student <PAD> <PAD>'
    hello_idx = torch.LongTensor([vocab[i] for i in text.split()])
    chinese_vocab = {'我': 0, '是': 1, '一个': 2, '学生': 3, '。': 4, '她': 5, '喜欢': 6, '跳舞': 7, '我们': 8, '吃': 9,
                     '比萨': 10, '天气': 11, '很': 12, '好': 13, '今天': 14, '<PAD>': 15}
    chinese_input_size = len(chinese_vocab)
    encoder = Encoder(chinese_input_size, hidden_dim)
    decoder = Decoder(hidden_dim, chinese_input_size)
    model = Seq2Seq(encoder, decoder)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()


    # 定义英文句子
    english_text = 'I am a student .'
    english_idx = torch.LongTensor([vocab[i] for i in english_text.split()])

    chinese_text = '我 是 学生'
    chinese_idx = torch.LongTensor([chinese_vocab[i] for i in chinese_text.split()])

    for epoch in range(epochs):
        loss = train(model, english_idx, chinese_idx, criterion, optimizer)
        print(f"Epoch: {epoch + 1}, Loss: {loss:.4f}")
    # 将英文句子输入编码器获取输出向量和隐藏状态
    #print(hidden.size())
    #print(output.size())
    #hidden = hidden[-decoder.num_layers:].unsqueeze(1)

    # 初始化解码器的输入，使用起始标记符号
    #start_symbol = torch.LongTensor([chinese_vocab['<PAD>']])  # 假设起始符号为<PAD>，可以根据实际情况修改
    #decoder_input = start_symbol


    #decoder_input = start_symbol.unsqueeze(1)

    #print(translated_sentence)


    '''
    translated_sentence = []
    #prediction = decoder(decoder_input, hidden)
    # print(prediction)
    predicted_word_index = torch.argmax(prediction, dim=2)
    # print(predicted_word_index)
    translated_word = list(chinese_vocab.keys())[predicted_word_index.item()]
    # print(list(chinese_vocab.keys()))

    translated_sentence.append(translated_word)
    decoder_input = predicted_word_index
    


    print(' '.join(translated_sentence))
    '''





    # 解码过程
    '''
    translated_sentence = []
    for _ in range(100):  # 定义生成的最大长度，这里使用max_length表示
        # 使用解码器进行解码
        decoder_input = start_symbol.unsqueeze(0)
        prediction, hidden = decoder(decoder_input.unsqueeze(0), hidden)

        # 选择预测概率最高的单词作为当前时间步的输出
        predicted_word_index = torch.argmax(prediction, dim=2)

        # 将预测的中文单词添加到结果列表中
        translated_word = list(chinese_vocab.keys())[predicted_word_index.item()]
        translated_sentence.append(translated_word)

        # 将当前时间步的输出作为下一个时间步的输入
        decoder_input = predicted_word_index

    # 打印翻译结果
    print(' '.join(translated_sentence))
'''

    # 创建编码器实例


    # 生成随机输入序列


    # 将输入序列传递给编码器
    #output, hidden = encoder(input_sequence)

    # 打印输出和隐藏状态的形状
    #print("Output shape:", output.shape)
    #print("Hidden shape:", hidden.shape)

# 运行测试函数
test_encoder()
"""