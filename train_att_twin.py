import sys,os
import numpy as np
import torch as t
import tqdm
import ipdb
import logging
import random

from models.lstm_att_twin import lstm_att_twin
from config import opt
from torchtext import data
from utils import Visualizer
from torchnet import meter
from torch import nn
from torch.autograd import Variable


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


def train_attention_twin(**kwargs):

    for k, v in kwargs.items():
        setattr(opt, k, v)
        # setattr(object, name, value) 设置属性值

    vis = Visualizer(env=opt.env)  # 设置visdom的环境变量

    logging.info("============attention_twin================")

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
        print('this is {0}'.format(count+1))
        model.train()
        loss_meter.reset()
        logging.info("this is the {0}th epoch".format(count + 1))
        cnt = 0

        for batch in tqdm.tqdm(train_iter):
            fwd_loss = 0

            # 训练
            data = batch.text
            batch_size = data.shape[1]
            seq_len = data.shape[0]
            idx = np.arange(seq_len)[::-1].tolist()
            idx = t.LongTensor(idx)
            idx = Variable(idx).cuda()

            att_hidden = Variable(t.zeros(batch_size, 150), requires_grad=False)  # (batch_size, hidden_dim)
            pre_hiddens = Variable(t.zeros(batch_size, 1, 150), requires_grad=False)
            f_h = None
            if opt.use_gpu:
                data = data.cuda()
                att_hidden = att_hidden.cuda()
                pre_hiddens = pre_hiddens.cuda()

            optimizer.zero_grad()
            # 输入和目标错开，CharRNN的做法
            input_, target_ = Variable(data[:-1, :]), Variable(data[1:, :])
            bx = data.index_select(0, idx)
            b_input, b_target = Variable(bx[:-1, :]), Variable(bx[1:, :])

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
                if ii == 0:
                    f_h = att_hidden.unsqueeze(0)
                else:
                    f_h = t.cat((f_h, att_hidden.unsqueeze(0)),0)
                # topv, topi = decoder_output.topk(1)
                # decoder_input = topi.squeeze().detach()  # detach from history as input
                fwd_loss += criterion(output, target)
            fwd_loss = fwd_loss / max_len

            # 反向网络
            b_out, b_h = model.bwd_forward(b_input, bwd_hidden)
            b_loss = criterion(b_out, b_target.view(-1))

            seq_len, batch_size, _ = f_h.size()
            # 计算twin_loss
            f_h = model.fwd_affine(f_h)
            b_h_inv = b_h.index_select(0, idx[1:])
            b_h_inv = b_h_inv[1:]  # 将<sos>去除
            # print(f_h.size(), b_h_inv.size())
            b_h_inv = b_h_inv.detach()
            f_h = f_h[:-1]  # 将<eos>去掉
            twin_loss = ((f_h - b_h_inv) ** 2).mean()
            twin_loss *= 1.5

            all_loss = b_loss + fwd_loss + twin_loss
            all_loss.backward()
            # 梯度剪裁
            t.nn.utils.clip_grad_norm(model.parameters(), 5.)
            optimizer.step()

            loss_meter.add(all_loss.item())

            # 可视化
            if (1 + cnt) % opt.plot_every == 0:
                # vis.plot('loss', loss_meter.value()[0])
                logging.info("train the %dth batch_plot's loss is: %f" % ((cnt+1)/opt.plot_every, loss_meter.value()[0]))
            cnt += 1

        count += 1

        valid_loss = evaluate_att(model, valid_iter, criterion)
        scheduler.step(valid_loss)
        logging.info("======the %dth validation's loss is: %f=====" % (count, valid_loss))
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model
            t.save(best_model.state_dict(), '%s%s_%d.pth' % (opt.model_prefix, opt.model, count))

        test_loss = evaluate_att(best_model, test_iter, criterion)
        logging.info("------test's loss为: %f" % test_loss)

        # 学习率减半
        if epoch in [5, 10, 15]:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                lr *= 0.5
                param_group['lr'] = lr

if __name__ == "__main__":
    import fire
    fire.Fire()