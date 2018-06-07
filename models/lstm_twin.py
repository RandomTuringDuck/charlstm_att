# coding:utf8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from config import opt


class lstm_twin(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(lstm_twin, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = None
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(opt.dropout, inplace=True)
        self.fwd_lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2, dropout=opt.dropout)
        self.bwd_lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2, dropout=opt.dropout)
        self.fwd_out = nn.Linear(self.hidden_dim, vocab_size)
        self.bwd_out = nn.Linear(self.hidden_dim, vocab_size)
        self.fwd_aff = nn.Linear(self.hidden_dim, self.hidden_dim)

    def init_hidden(self):
        # Define the hidden state and the cell, the parameter 2's meaning is two layers of LSTM.
        if opt.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def work(self, input, hidden, forward=True):
        lstm = self.fwd_lstm if forward else self.bwd_lstm
        out = self.fwd_out if forward else self.bwd_out
        seq_len, batch_size = input.size()
        h_0, c_0 = hidden
        # embeds's size: (seq_len,batch_size,embeding_dim)
        embeds = self.embeddings(input)
        embeds = self.dropout(embeds)
        # output's size: (seq_len,batch_size,hidden_dim)
        output, hidden = lstm(embeds, (h_0, c_0))
        vis = output
        if forward:
            vis_ = self.fwd_aff(output.view(seq_len * batch_size, self.hidden_dim))
            vis = vis_.view(seq_len, batch_size, self.hidden_dim)

        output = out(output.view(seq_len * batch_size, -1))
        # 最后得到每个字的候选下一个字的概率，候选概率共有vocab_size个，现在要
        # 求交叉熵损失，target的size为(seq_len*batch_size).
        return output, vis, hidden

    def forward(self, f_input, b_input, hidden1, hidden2):
        fwd_out, fwd_h, _ = self.work(f_input, hidden1)
        bwd_out, bwd_h, _ = self.work(b_input, hidden2, forward=False)
        return fwd_out, bwd_out, fwd_h, bwd_h



