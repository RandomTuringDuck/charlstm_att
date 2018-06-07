# coding:utf8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from config import opt


class lstm(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(lstm, self).__init__()
        self.dropout = opt.dropout
        self.hidden_dim = hidden_dim
        '''
            Embedding函数的两个必要参数
            num_embeddings (int) – size of the dictionary of embeddings
            embedding_dim (int) – the size of each embedding vector,每个词的embedding vector的size

            input:LongTensor (N, W), N = mini-batch, W = number of indices to extract per mini-batch
            Output: (N, W, embedding_dim)，即将矩阵的每个点给你用size为embedding_dim的向量表示一下。
        '''
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        '''
            LSTM函数初始化时需要三个必要参数，第一个是输入词的维度input_size。
            第二个是隐藏元的维度,hidden_size
            第三个是LSTM的层数,num_layers

            Inputs： input,h_0,c_0
                input(seq_len, batch, input_size）,embeddings的结果为(seq_len,batch_size,embeding_dim),lstm初始化时的input_size为embeding_dim，
                即每个词的特征维度
                h_0 (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
                c_0 (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for each element in the batch.
        '''
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2, dropout=self.dropout)
        '''
            输入为(N,input_features),
            输出为(N,output_features).
        '''
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)  # the first parameter is in_features，the following is out_features.

    def forward(self, input, hidden=None):
        '''
        这里的input的size是(seq_len,batch_size),即每一列是一首诗，
        列数是batch_size。
        这个项目里seq_len为124，batch_size为128,可以指定batch_size。
        '''
        seq_len, batch_size = input.size()
        if hidden is None:
            #  h_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            #  c_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            #  定义隐藏状态单元和cell状态,2是因为是两层LSTM
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            h_0, c_0 = Variable(h_0), Variable(c_0)
        else:
            h_0, c_0 = hidden
        # embeds's size: (seq_len,batch_size,embeding_dim)
        embeds = self.embeddings(input)
        # output's size: (seq_len,batch_size,hidden_dim)
        output, hidden = self.lstm(embeds, (h_0, c_0))

        # input size: (seq_len*batch_size,hidden_dim)
        # output size: (seq_len*batch_size,vocab_size)
        output = self.linear1(output.view(seq_len * batch_size, -1))
        # 最后得到每个字的候选下一个字的概率，候选概率共有vocab_size个，现在要
        # 求交叉熵损失，target的size为(seq_len*batch_size).
        return output, hidden



