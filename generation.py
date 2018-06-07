#coding:utf8
import torch as t
from models.lstm import lstm
from models.attention import lstm_att
from models.lstm_twin import lstm_twin
from config import opt
from torchtext import data
from torch.autograd import Variable
import logging
import os
import numpy as np

logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename=opt.logging_file,
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )

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

def gen_comment(**kwargs):
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
    print("加载完毕")

    start_word = ['用这个牌子好多年了','篮球手感','蛋白质粉','价格','什么也不想说了','有用过这个牌子',
                  '箱子外观很漂亮','家里人','店家','比我预料中的好']
    if opt.use_gpu:
        model.cuda()

    hidden = None
    comments = []

    for ii in start_word:
        result = list(ii)
        input = Variable(t.Tensor([word2ix['<start>']]).view(1, 1).long())
        if opt.use_gpu: input = input.cuda()

        if opt.model == 'lstm_twin':
            model.batch_size = 1
            hidden = model.init_hidden()

        for i in range(opt.max_gen_len):
            if opt.model == 'lstm':
                output, hidden = model(input, hidden)
            elif opt.model == 'lstm_twin':
                output = model.work(input,hidden)
                hidden = output[2]
                output = output[0]


            if i < len(ii):
                w = result[i]
                input = Variable(input.data.new([word2ix[w]])).view(1, 1)
            else:
                top_index = output.data[0].topk(1)[1][0]
                w = ix2word[top_index]
                result.append(w)
                input = Variable(input.data.new([top_index])).view(1, 1)
            if w == '<eos>':
                del result[-1]
                break

        comments.append(result)

    print("打印评论：")
    for i in comments:
        print(''.join(i))

def gen_comment_att(**kwargs):
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

    model = lstm_att(len(word2ix), 300, 150)


    map_location = lambda s, l: s
    state_dict = t.load(opt.model_path, map_location=map_location)
    model.load_state_dict(state_dict)
    print("加载完毕")

    start_word = ['用这个牌子好多年了', '篮球手感', '蛋白质粉', '价格', '什么也不想说了', '听说写到']
    if opt.use_gpu:
        model.cuda()

    comments = []

    for ii in start_word:
        result = list(ii)
        input = Variable(t.Tensor([word2ix['<start>']]).long())
        batch_size = 1
        att_hidden = Variable(t.zeros(batch_size, 150))  # (batch_size, hidden_dim)
        pre_hiddens = Variable(t.zeros(batch_size, 1, 150))
        model.batch_size = batch_size
        hidden = model.init_hidden()
        if opt.use_gpu:
            input = input.cuda()
            att_hidden = att_hidden.cuda()
            pre_hiddens = pre_hiddens.cuda()

        for i in range(opt.max_gen_len):

            output, att_hidden, pre_hidden, hidden, alpha = model(input, att_hidden, pre_hiddens, hidden)
            logging.info("第%d次: %s" % (i, alpha))
            pre_hidden = pre_hidden.detach()
            pre_hiddens = t.cat((pre_hiddens, pre_hidden), 1)

            if i < len(ii):
                w = result[i]
                input = Variable(input.data.new([word2ix[w]]))
            else:
                top_index = output.data[0].topk(1)[1][0]
                w = ix2word[top_index]
                result.append(w)
                input = Variable(input.data.new([top_index]))
            if w == '<eos>':
                del result[-1]
                break

        comments.append(result)

    print("打印评论：")
    for i in comments:
        print(''.join(i))


if __name__ == "__main__":
    import fire
    fire.Fire()
