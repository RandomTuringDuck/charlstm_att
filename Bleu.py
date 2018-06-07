#coding:utf8
import sys,os
import numpy as np
import torch as t
import tqdm
import ipdb
import logging
import random

from models.lstm import lstm
from models.attention import lstm_att
from models.lstm_att_twin import lstm_att_twin
from models.lstm_twin import lstm_twin
from config import opt
from torchtext import data
from utils import Visualizer
from torchnet import meter
from torch import nn
from torch.autograd import Variable
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu

def load_data():
    # 如果要处理中文，就给它个tokenize.
    text_field = data.Field(tokenize=lambda x: list(x),init_token='<start>',eos_token='<eos>')
    #
    train, valid, test = data.TabularDataset.splits(path=opt.data_path,train='train.csv',validation='valid.csv',
                                       test = 'test.csv',format='csv',skip_header=True,fields=[("text", text_field)])

    text_field.build_vocab(train, valid, test)

    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        datasets=(train, valid, test), batch_size=opt.batch_size,
        sort_key=lambda x: len(x.text), repeat=False)

    return train_iter, valid_iter, test_iter, text_field

def Bleu(**kwargs):

    for k, v in kwargs.items():
        setattr(opt, k, v)
        # setattr(object, name, value) 设置属性值
    print('Loading model from {}'.format(opt.model_path))
    # 加载词典
    if os.path.exists(opt.pickle_path):
        data = np.load(opt.pickle_path)
        word2ix, ix2word = data['word2ix'].item(), data['ix2word']
    else:
        train_iter, valid_iter, test_iter, field = load_data()
        word2ix = field.vocab.stoi
        ix2word = field.vocab.itos
    # 加载模型
    if opt.model == 'lstm':
        model = lstm(len(word2ix), 300, 150)
    elif opt.model == 'lstm_twin':
        model = lstm_twin(len(word2ix), 300, 150)

    map_location = lambda s, l: s
    state_dict = t.load(opt.model_path, map_location=map_location)
    model.load_state_dict(state_dict)
    if opt.use_gpu:
        model.cuda()
    print("加载完毕")

    # model.eval()
    hypothesis=[]
    references = []
    cnt = 0
    for batch in tqdm.tqdm(test_iter):
        cnt += 1
    # batch = next(iter(test_iter))
        data = batch.text
        if opt.model == 'lstm_twin':
            model.batch_size = data.size(1)
            hidden = model.init_hidden()
        if opt.use_gpu:
            data = data.cuda()
        input_, target = Variable(data[:-1, :]), Variable(data[1:, :])
        tmp = target.transpose(0,1).cpu().numpy()
        # print(tmp)
        print('===========输入==========')
        for ii in tmp:
             ii_ = list(ii)
             for i in ii_:
                 print(ix2word[i], end='')
             print('')
             ii_ = ii_[:ii_.index(3)+1]
             references.append([ii_])

        print('===========输出==========')
        # print(references)

        if opt.model == 'lstm':
            output, _ = model(input_)
            output = output.view(data.size(0) - 1, data.size(1), -1)
        elif opt.model == 'lstm_twin':
            output = model.work(input_, hidden)
            output = output[0].view(data.size(0) - 1, data.size(1), -1)

        # print(output.size())
        top = output.topk(1,dim=2)[1].squeeze().transpose(0,1)
        top = top.cpu().numpy()
        for ii in top:
            ii_ = list(ii)
            for i in ii_:
                print(ix2word[i],end='')
            print('')
            haha = ii_.index(3) if 3 in ii_ else None
            if(haha):
                ii_ = ii_[:haha + 1]
            hypothesis.append(ii_)

        # if cnt > 10:
        #     break

        # print(hypothesis)
    bleu1 = corpus_bleu(references, hypothesis, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypothesis, weights=(1. / 2., 1. / 2., 0, 0))
    bleu3 = corpus_bleu(references, hypothesis, weights=(1. / 3., 1. / 3., 1. / 3., 0))
    bleu4 = corpus_bleu(references, hypothesis)
    print("bleu1: ", bleu1, "bleu2: ", bleu2, "bleu3: ", bleu3, "bleu4: ", bleu4)

    # return bleu1, bleu2, bleu3, bleu4

def Bleu_att(**kwargs):

    for k, v in kwargs.items():
        setattr(opt, k, v)
        # setattr(object, name, value) 设置属性值
    print('Loading model from {}'.format(opt.model_path))
    # 加载词典
    if os.path.exists(opt.pickle_path):
        data = np.load(opt.pickle_path)
        word2ix, ix2word = data['word2ix'].item(), data['ix2word'].item()
    else:
        train_iter, valid_iter, test_iter, field = load_data()
        word2ix = field.vocab.stoi
        ix2word = field.vocab.itos
    # 加载模型
    model = lstm_att(len(word2ix), 300, 150)

    map_location = lambda s, l: s
    state_dict = t.load(opt.model_path, map_location=map_location)
    model.load_state_dict(state_dict)
    if opt.use_gpu:
        model.cuda()
    print("加载完毕")

    # model.eval()
    hypothesis=[]
    references = []

    for batch in tqdm.tqdm(test_iter):
        # batch = next(iter(test_iter))
        data = batch.text
        batch_size = data.shape[1]
        att_hidden = Variable(t.zeros(batch_size, 150))  # (batch_size, hidden_dim)
        pre_hiddens = Variable(t.zeros(batch_size, 1, 150))

        if opt.use_gpu:
            data = data.cuda()
            att_hidden = att_hidden.cuda()
            pre_hiddens = pre_hiddens.cuda()

        input_, target_ = Variable(data[:-1, :]), Variable(data[1:, :])
        tmp = target_.transpose(0, 1).cpu().numpy()
        # print(tmp)
        for ii in tmp:
             ii_ = list(ii)
             ii_ = ii_[:ii_.index(3)+1]
             references.append([ii_])

        # print(references)

        max_len = input_.size(0)
        model.batch_size = batch_size
        hidden = model.init_hidden()

        hy = None
        for ii in range(max_len):
            input = input_[ii]  # (batch_size,)
            output, att_hidden, pre_hidden, hidden, alpha = model(input, att_hidden, pre_hiddens, hidden)
            pre_hidden = pre_hidden.detach()
            pre_hiddens = t.cat((pre_hiddens, pre_hidden), 1)
            tmp = output.topk(1, dim=1)[1].cpu().numpy()
            if ii == 0:
                hy = tmp.copy()
            else:
                hy = np.append(hy,tmp,axis=1)

        for ii in hy:
            ii_ = list(ii)
            haha = ii_.index(3) if 3 in ii_ else None
            if(haha):
                ii_ = ii_[:haha + 1]
            hypothesis.append(ii_)

    bleu1 = corpus_bleu(references, hypothesis, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypothesis, weights=(1. / 2., 1. / 2., 0, 0))
    bleu3 = corpus_bleu(references, hypothesis, weights=(1. / 3., 1. / 3., 1. / 3., 0))
    bleu4 = corpus_bleu(references, hypothesis)
    print("bleu1: ", bleu1, "bleu2: ", bleu2, "bleu3: ", bleu3, "bleu4: ", bleu4)

    # return bleu1, bleu2, bleu3, bleu4


if __name__ == "__main__":
    import fire
    fire.Fire()