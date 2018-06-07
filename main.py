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


MAX_LENGTH = 100
teacher_forcing_ratio = 0.5

logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename=opt.logging_file,
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )

# 获得三个数据集的迭代器
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


def evaluate(model, data_iter, loss_function):
    model.eval()
    loss_meter = meter.AverageValueMeter()
    loss_meter.reset()
    for batch in tqdm.tqdm(data_iter):
        data = batch.text
        if opt.use_gpu:
            data = data.cuda()
        input_, target = Variable(data[:-1, :]), Variable(data[1:, :])
        output,_ = model(input_)
        loss = loss_function(output, target.view(-1))
        loss_meter.add(loss.data[0])

    return loss_meter.value()[0]

# without attention
def train(**kwargs):

    for k, v in kwargs.items():
        setattr(opt, k, v)
        # setattr(object, name, value) 设置属性值

    vis = Visualizer(env=opt.env)  # 设置visdom的环境变量

    # 获取数据
    train_iter, valid_iter, test_iter, field = load_data()
    word2ix = field.vocab.stoi
    ix2word = field.vocab.itos

    # np.savez('data/word2ix.npz', word2ix = word2ix,ix2word = ix2word)

    # 模型定义
    model = lstm(len(word2ix), 300, 150)

    best_model = model
    best_valid_loss = float("inf")

    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()

    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))

    if opt.use_gpu:
        model.cuda()
        criterion.cuda()

    loss_meter = meter.AverageValueMeter()
    count = 0
    for epoch in range(opt.epoch):
        model.train()
        loss_meter.reset()
        logging.info("这是第{0}次epoch".format(count + 1))
        cnt = 0
        for batch in tqdm.tqdm(train_iter):  # tqdm是一个python进度条库，可以封装iterator，it/s表示的就是每秒迭代了多少次
            # 训练
            data = batch.text
            if opt.use_gpu: data = data.cuda()
            optimizer.zero_grad()

            # 输入和目标错开，CharRNN的做法
            input_, target = Variable(data[:-1, :]), Variable(data[1:, :])
            output, _ = model(input_)

            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())

            # 可视化
            if (1 + cnt) % opt.plot_every == 0:
                vis.plot('loss', loss_meter.value()[0])
            cnt += 1
        count += 1

        valid_loss = evaluate(model, valid_iter, criterion)
        logging.info("第%d次验证集的loss为: %f"%(count, valid_loss))
        if valid_loss < best_valid_loss:
            os.system('rm ' + opt.model_prefix +opt.model + '.pth')
            best_valid_loss = valid_loss
            best_model = model
            t.save(best_model.state_dict(), '%s%s.pth' % (opt.model_prefix, opt.model))

        test_loss = evaluate(best_model,test_iter,criterion)
        logging.info("测试集的loss为: %f" % test_loss)

def evaluate_twin(model, data_iter, loss_function):
    model.eval()

    loss_meter = meter.AverageValueMeter()
    loss_meter.reset()

    for batch in tqdm.tqdm(data_iter):
        data = batch.text
        model.batch_size = data.size(1)
        hidden = model.init_hidden()
        if opt.use_gpu:
            data = data.cuda()
        input_, target = Variable(data[:-1, :]), Variable(data[1:, :])
        output = model.work(input_, hidden)

        loss = loss_function(output[0], target.view(-1))
        loss_meter.add(loss.item())

    return loss_meter.value()[0]

# without attention
def train_twin(**kwargs):

    for k, v in kwargs.items():
        setattr(opt, k, v)
        # setattr(object, name, value) 设置属性值

    vis = Visualizer(env=opt.env)  # 设置visdom的环境变量

    # 获取数据
    train_iter, valid_iter, test_iter, field = load_data()
    word2ix = field.vocab.stoi
    ix2word = field.vocab.itos
    # 模型定义
    model = lstm_twin(len(word2ix), 300, 150)

    best_model = model
    best_valid_loss = float("inf")

    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5)
    # CrossEntropyLoss 会把每个字符的损失求平均，所以损失是个10以内的数，如果加上size_average = False, 就变成一个10000以内的
    # 数了，正好差不多2000倍吧，如果想以每句话为单位，那么就乘上seq_len
    criterion = nn.CrossEntropyLoss()

    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))

    if opt.use_gpu:
        model.cuda()
        criterion.cuda()

    count = 0
    for epoch in range(opt.epoch):
        model.train()
        logging.info("这是第{0}次epoch".format(count + 1))
        cnt = 0

        b_fwd_loss, b_bwd_loss, b_twin_loss, b_all_loss = 0., 0., 0., 0.

        for batch in tqdm.tqdm(train_iter):  # tqdm是一个python进度条库，可以封装iterator，it/s表示的就是每秒迭代了多少次
            # 训练
            data = batch.text
            seq_len = data.size(0)
            # 生成一个倒着的序列,因为tensor不支持负步长
            idx = np.arange(seq_len)[::-1].tolist()
            idx = t.LongTensor(idx)
            idx = Variable(idx).cuda()
            model.batch_size = data.size(1)
            hidden1 = model.init_hidden()
            hidden2 = model.init_hidden()
            if opt.use_gpu: data = data.cuda()
            optimizer.zero_grad()

            # 输入和目标错开，CharRNN的做法
            f_input, f_target = Variable(data[:-1, :]), Variable(data[1:, :])
            bx = data.index_select(0, idx)
            b_input, b_target = Variable(bx[:-1,:]), Variable(bx[1:,:])
            # print(f_input.size(),b_input.size())
            f_out, b_out, f_h, b_h = model(f_input, b_input, hidden1, hidden2)

            f_loss = criterion(f_out, f_target.view(-1))
            b_loss = criterion(b_out, b_target.view(-1))
            b_h_inv = b_h.index_select(0, idx[1:])
            b_h_inv = b_h_inv[1:] #将<sos>去除
            # print(f_h.size(), b_h_inv.size())
            b_h_inv = b_h_inv.detach()
            f_h = f_h[:-1] #将<eos>去掉
            twin_loss = ((f_h - b_h_inv) ** 2).mean()
            twin_loss *= 1.5
            all_loss = f_loss + b_loss + twin_loss
            all_loss.backward()

            t.nn.utils.clip_grad_norm(model.parameters(), 5.)
            optimizer.step()

            # 累加
            b_all_loss += all_loss.item()
            b_fwd_loss += f_loss.item()
            b_bwd_loss += b_loss.item()
            b_twin_loss += twin_loss.item()

            # 可视化
            if (1 + cnt) % opt.plot_every == 0:
                vis.plot('all_loss', b_all_loss/opt.plot_every)
                vis.plot('twin_loss', b_twin_loss / opt.plot_every)
                vis.plot('loss', b_fwd_loss/opt.plot_every)
                # logging.info("训练第{}个plot的all_loss:{:f}, f_loss: {:f}, b_loss: {:f}, twin_loss: {:f}"
                #              .format(int((cnt + 1) / opt.plot_every), b_all_loss / opt.plot_every,
                #                      b_fwd_loss / opt.plot_every,
                #                      b_bwd_loss / opt.plot_every, b_twin_loss / opt.plot_every))

                b_fwd_loss, b_bwd_loss, b_twin_loss, b_all_loss = 0., 0., 0., 0.

            cnt += 1
        count += 1

        valid_loss = evaluate_twin(model, valid_iter, criterion)
        scheduler.step(valid_loss)
        logging.info("第%d次验证集的loss为: %f"%(count, valid_loss))
        if valid_loss < best_valid_loss:
            # os.system('rm ' + opt.model_prefix +opt.model + '.pth')
            best_valid_loss = valid_loss
            best_model = model
            t.save(best_model.state_dict(), '%s%s_%d.pth' % (opt.model_prefix, opt.model, count))

        test_loss = evaluate_twin(best_model,test_iter,criterion)
        logging.info("测试集的loss为: %f" % test_loss)

        # 学习率减半
        if epoch in [5, 10, 15]:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                lr *= 0.5
                param_group['lr'] = lr



# with attention
def evaluate_att(model, data_iter, loss_function):
    model.eval()
    loss_meter = meter.AverageValueMeter()
    loss_meter.reset()
    for batch in tqdm.tqdm(data_iter):
        loss = 0
        data = batch.text
        batch_size = data.shape[1]
        att_hidden = Variable(t.zeros(batch_size, 150))  # (batch_size, hidden_dim)
        pre_hiddens = Variable(t.zeros(batch_size, 1, 150))

        if opt.use_gpu:
            data = data.cuda()
            att_hidden = att_hidden.cuda()
            pre_hiddens = pre_hiddens.cuda()

        input_, target_ = Variable(data[:-1, :]), Variable(data[1:, :])
        max_len = input_.size(0)
        model.batch_size = batch_size
        hidden = model.init_hidden()

        for ii in range(max_len):
            input = input_[ii]  # (batch_size,)
            target = target_[ii]
            output, att_hidden, pre_hidden, hidden, alpha = model(input, att_hidden, pre_hiddens, hidden)
            pre_hidden = pre_hidden.detach()
            pre_hiddens = t.cat((pre_hiddens, pre_hidden), 1)
            loss += loss_function(output, target)

        loss_meter.add(loss.item()/max_len)

    return loss_meter.value()[0]


# with attention
def train_attention(**kwargs):

    for k, v in kwargs.items():
        setattr(opt, k, v)
        # setattr(object, name, value) 设置属性值

    vis = Visualizer(env=opt.env)  # 设置visdom的环境变量

    logging.info("============attention的训练过程================")

    # 获取数据
    train_iter, valid_iter, test_iter, field = load_data()
    word2ix = field.vocab.stoi
    ix2word = field.vocab.itos

    # 模型定义
    model = lstm_att(len(word2ix), 300, 150)

    best_model = model
    best_valid_loss = float("inf")

    # lambda1 = lambda epoch: epoch // 5
    # lambda2 = lambda epoch: 0.95 ** epoch

    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-6)
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = nn.NLLLoss()

    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))

    if opt.use_gpu:
        model.cuda()
        criterion.cuda()

    loss_meter = meter.AverageValueMeter()
    count = 0
    for epoch in range(opt.epoch):

        model.train()
        loss_meter.reset()
        logging.info("这是第{0}次epoch".format(count + 1))
        cnt = 0

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        for batch in tqdm.tqdm(train_iter):
            loss = 0
            # 训练
            data = batch.text
            batch_size = data.shape[1]
            att_hidden = Variable(t.zeros(batch_size, 150), requires_grad=False)  # (batch_size, hidden_dim)
            pre_hiddens = Variable(t.zeros(batch_size, 1, 150), requires_grad=False)
            if opt.use_gpu:
                data = data.cuda()
                att_hidden = att_hidden.cuda()
                pre_hiddens = pre_hiddens.cuda()

            optimizer.zero_grad()
            # 输入和目标错开，CharRNN的做法
            input_, target_ = Variable(data[:-1, :]), Variable(data[1:, :])
            max_len = input_.size(0)
            model.batch_size = batch_size
            hidden = model.init_hidden()

            for ii in range(max_len):
                input = input_[ii]  # (batch_size,)
                target = target_[ii]
                output, att_hidden, pre_hidden, hidden, alpha = model(input, att_hidden, pre_hiddens, hidden)
                # logging.info("第%d次: %s" % (ii, alpha))
                pre_hidden = pre_hidden.detach()
                pre_hiddens = t.cat((pre_hiddens, pre_hidden), 1)
                # topv, topi = decoder_output.topk(1)
                # decoder_input = topi.squeeze().detach()  # detach from history as input
                loss += criterion(output, target)

            loss.backward()
            # 梯度剪裁
            t.nn.utils.clip_grad_norm(model.parameters(), 5.)
            optimizer.step()

            loss_meter.add(loss.item()/max_len)

            # 可视化
            if (1 + cnt) % opt.plot_every == 0:
                vis.plot('loss', loss_meter.value()[0])
                # logging.info("训练第%d次batch_plot的loss为: %f" % ((cnt+1)/opt.plot_every, loss_meter.value()[0]))
            cnt += 1

        count += 1

        valid_loss = evaluate_att(model, valid_iter, criterion)
        scheduler.step(valid_loss)
        logging.info("======第%d次验证集的loss为: %f=====" % (count, valid_loss))
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model
            t.save(best_model.state_dict(), '%s%s_%d.pth' % (opt.model_prefix, opt.model, count))

        test_loss = evaluate_att(best_model, test_iter, criterion)
        logging.info("------测试集的loss为: %f" % test_loss)

        # 学习率减半
        if epoch in [5, 10, 15]:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                lr *= 0.5
                param_group['lr'] = lr

# with attention
def train_attention_twin(**kwargs):

    for k, v in kwargs.items():
        setattr(opt, k, v)
        # setattr(object, name, value) 设置属性值

    vis = Visualizer(env=opt.env)  # 设置visdom的环境变量

    logging.info("============attention_twin的训练过程================")

    # 获取数据
    train_iter, valid_iter, test_iter, field = load_data()
    word2ix = field.vocab.stoi
    ix2word = field.vocab.itos

    # 模型定义
    model = lstm_att_twin(len(word2ix), 300, 150)

    best_model = model
    best_valid_loss = float("inf")

    # lambda1 = lambda epoch: epoch // 5
    # lambda2 = lambda epoch: 0.95 ** epoch

    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-6)
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = nn.NLLLoss()

    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))

    if opt.use_gpu:
        model.cuda()
        criterion.cuda()

    loss_meter = meter.AverageValueMeter()
    count = 0
    for epoch in range(opt.epoch):

        model.train()
        loss_meter.reset()
        logging.info("这是第{0}次epoch".format(count + 1))
        cnt = 0

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        for batch in tqdm.tqdm(train_iter):
            loss = 0
            # 训练
            data = batch.text
            batch_size = data.shape[1]
            att_hidden = Variable(t.zeros(batch_size, 150), requires_grad=False)  # (batch_size, hidden_dim)
            pre_hiddens = Variable(t.zeros(batch_size, 1, 150), requires_grad=False)
            if opt.use_gpu:
                data = data.cuda()
                att_hidden = att_hidden.cuda()
                pre_hiddens = pre_hiddens.cuda()

            optimizer.zero_grad()
            # 输入和目标错开，CharRNN的做法
            input_, target_ = Variable(data[:-1, :]), Variable(data[1:, :])
            max_len = input_.size(0)
            model.batch_size = batch_size
            hidden = model.init_hidden()
            bwd_hidden = model.init_hidden()

            for ii in range(max_len):
                input = input_[ii]  # (batch_size,)
                target = target_[ii]
                output, att_hidden, pre_hidden, hidden, alpha = model(input, att_hidden, pre_hiddens, hidden)
                # logging.info("第%d次: %s" % (ii, alpha))
                pre_hidden = pre_hidden.detach()
                pre_hiddens = t.cat((pre_hiddens, pre_hidden), 1)
                # topv, topi = decoder_output.topk(1)
                # decoder_input = topi.squeeze().detach()  # detach from history as input
                loss += criterion(output, target)

            loss.backward()
            # 梯度剪裁
            t.nn.utils.clip_grad_norm(model.parameters(), 5.)
            optimizer.step()

            loss_meter.add(loss.item()/max_len)

            # 可视化
            if (1 + cnt) % opt.plot_every == 0:
                vis.plot('loss', loss_meter.value()[0])
                # logging.info("训练第%d次batch_plot的loss为: %f" % ((cnt+1)/opt.plot_every, loss_meter.value()[0]))
            cnt += 1

        count += 1

        valid_loss = evaluate_att(model, valid_iter, criterion)
        scheduler.step(valid_loss)
        logging.info("======第%d次验证集的loss为: %f=====" % (count, valid_loss))
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model
            t.save(best_model.state_dict(), '%s%s_%d.pth' % (opt.model_prefix, opt.model, count))

        test_loss = evaluate_att(best_model, test_iter, criterion)
        logging.info("------测试集的loss为: %f" % test_loss)

        # 学习率减半
        if epoch in [5, 10, 15]:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                lr *= 0.5
                param_group['lr'] = lr


if __name__ == "__main__":
    import fire
    fire.Fire()