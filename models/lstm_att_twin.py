# coding:utf8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from config import opt


class Attention(nn.Module):
    def __init__(self, method, hidden_dim):
        super(Attention,self).__init__()
        self.method = method
        self.hidden_dim = hidden_dim

        if self.method == 'general':
            self.attn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif self.method == 'concat':
            self.attn = nn.Linear(hidden_dim*2, hidden_dim, bias=False)
            self.tanh = nn.Tanh()
            self.attn_linear = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, pre_hiddens):
        """
        :param hidden: decode hidden state, (batch_size , hidden_dim)
        :param pre_hiddens: time 1~t-1 hidden state, (batch_size,T-1,hidden_dim)
        :return: context :(batch_size,hidden_dim), alpha:(batch_size,T-1)
        """
        hidden_expanded = hidden.unsqueeze(2)  # (batch_size,N,1)

        # 使用bmm第一维一定要一样
        if self.method == 'dot':
            score = torch.bmm(pre_hiddens, hidden_expanded).squeeze(2)  # (T-1,N) * (N,1)

        elif self.method == 'general':
            score = self.attn(pre_hiddens)
            score = torch.bmm(score, hidden_expanded).squeeze(2)

        elif self.method == 'concat':
            hidden_expanded = hidden.unsqueeze(1).expand_as(pre_hiddens)
            score = self.attn(torch.cat((hidden_expanded, pre_hiddens), 2))  # （batch_size, T-1, hidden_dim）
            score = self.attn_linear(self.tanh(score)).squeeze(2)  # (batch_size, T-1)

        alpha = nn.functional.softmax(score, dim=1)  # (batch_size, T-1)
        context = torch.bmm(alpha.unsqueeze(1), pre_hiddens).squeeze(1)  # (1,T-1) * (T-1, hidden_dim)

        return context, alpha


class lstm_att_twin(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(lstm_att_twin, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = None
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(opt.dropout, inplace=True)
        # lstm,用h'_(t-1)+input来生成h_t.
        self.lstm = nn.LSTM(embedding_dim+hidden_dim, self.hidden_dim, num_layers=3, dropout=opt.dropout)
        # attention
        self.attention = Attention('concat', self.hidden_dim)
        # 利用c_t和h_t去预测一个h'_t,然后用这个h‘_t去生成输出
        self.out1 = nn.Linear(self.hidden_dim*2,self.hidden_dim)
        self.tanh = nn.Tanh()
        self.out2 = nn.Linear(self.hidden_dim, vocab_size)  #输入是h'_t

        self.bwd_lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=3, dropout=opt.dropout)
        self.bwd_out = nn.Linear(self.hidden_dim, vocab_size)
        self.fwd_aff = nn.Linear(self.hidden_dim, self.hidden_dim)

    def init_hidden(self):
        #Define the hidden state and the cell, the parameter 2's meaning is two layers of LSTM.
        if opt.use_gpu:
            return (Variable(torch.zeros(3, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(3, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(3, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(3, self.batch_size, self.hidden_dim)))

    def bwd_forward(self, input, hidden):
        embeds = self.embeddings(input)
        embeds = self.dropout(embeds)
        # output's size: (seq_len,batch_size,hidden_dim)
        seq_len, batch_size = input.size()
        h_0, c_0 = hidden
        output, hidden = self.bwd_lstm(embeds, (h_0, c_0))
        vis = output
        output = self.bwd_out(output.view(seq_len * batch_size, -1))
        output = F.log_softmax(output, dim=1)
        return output, vis

    def fwd_affine(self, input):
        print('wocao',input.size())
        seq_len, batch_size, _ = input.size()
        vis_ = self.fwd_aff(input.view(seq_len * batch_size, self.hidden_dim))
        vis = vis_.view(seq_len, batch_size, self.hidden_dim)
        return vis

    def forward(self, input, att_hidden, pre_hiddens, hidden):
        '''
        input's size (1,batch_size)
        att_hidden 最后用h_t和c_t一起得到的状态 （batch_size,）
        pre_hiddens [1,t-1]的所有的hidden states, （t-1, batch_size, hidden_dim）
        hidden 本身lstm要用的hidden,不用管
        '''
        h_0, c_0 = hidden
        embeds = self.embeddings(input)   # embeds's size: (batch_size, embedding_dim)
        embeds = self.dropout(embeds)
        # print(embeds.size(), att_hidden.size())
        lstm_input = torch.cat((embeds, att_hidden), 1)    # (batch_size, embedding_dim+hidden_dim)
        output, hidden = self.lstm(lstm_input.unsqueeze(0), (h_0, c_0))    # (1, batch_size, hidden_dim)
        # 因为是一个字一个字往里喂的，所以output只有一个hidden state
        pre_hidden = output.transpose(0,1)     # pre_hidden是上一次状态的最后一层隐藏元的值，传回去
        output = output.view(self.batch_size, self.hidden_dim)  # (batch_size, hidden_dim)
        # print(output.size())
        # print(pre_hiddens.size())
        context, alpha = self.attention(output, pre_hiddens)  # (batch_size, hidden_dim)
        att_hidden = self.tanh(self.out1(torch.cat((output, context), 1)))
        output = self.out2(att_hidden)
        output = F.log_softmax(output, dim=1)
        return output, att_hidden, pre_hidden, hidden, alpha



