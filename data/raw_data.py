import jieba
import html
import random
from urllib.parse import unquote
from torchtext import data
import csv
import numpy as np
import torchtext
from torchtext import data
from config import opt
import os

'''
用来处理一开始的raw_data的，用完之后的程序就用不到了
'''
def raw_to_train(filename):

    f = open(filename,'r')
    out = open('data/train_data.txt','w')
    result = []
    for line in f.readlines():
        tmp = line.split('\t')[0]
        tmp = html.unescape(tmp)
        tmp = unquote(tmp, 'utf-8')
        out.write(tmp+'\n')

    f.close()
    out.close()

def split_data(filename):
    f = open(filename,'r')
    result = []
    for line in f.readlines():
        tmp = line.strip()
        if len(tmp) > 100 or len(tmp) < 15:
            print(tmp)
            continue
        result.append(tmp)
    print(result.__len__())

    random.shuffle(result)

    train = result[:200000]
    valid = result[200001:240000]
    test = result[240001:]

    # for i in test:
    #     tmp = []
    #     tmp.append(i)
    #     print(tmp)
    out = open('data/train.csv','w', newline='')
    writer = csv.writer(out)
    for i in train:
        tmp = []
        tmp.append(i)
        writer.writerow(tmp)
    out.close()
    out = open('data/valid.csv','w',newline='')
    writer = csv.writer(out)
    for i in valid:
        tmp = []
        tmp.append(i)
        writer.writerow(tmp)
    out.close()
    out = open('data/test.csv','w',newline='')
    writer = csv.writer(out)
    for i in test:
        tmp = []
        tmp.append(i)
        writer.writerow(tmp)
    out.close()

    f.close()



def segmentation(filename):
    result = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            seg = jieba.cut(line.strip(), cut_all=False)
            for ii in seg:
                result.append(ii)
        result = set(result)

    print(result.__len__())


def load_data():
    # 如果要处理中文，就给它个tokenize.
    text_field = data.Field(tokenize=lambda x: list(x),init_token='<start>',eos_token='<eos>')
    #
    train, valid, test = data.TabularDataset.splits(path=opt.data_path,train='train.csv',validation='valid.csv',
                                       test = 'test.csv',format='csv',skip_header=True,fields=[("text", text_field)])

    text_field.build_vocab(train, valid, test)

    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train, valid, test), batch_size=opt.batch_size,
        sort_key=lambda x: len(x.text), repeat=False)

    return train_iter, valid_iter, test_iter, text_field


trn,vaild,test,vocab= load_data()
# print(trn.__len__())
# # for batch in trn:
# #     print(batch.text.data.shape)
# #     # if batch.text.data.shape[0]>1000:
# #     #     # print(batch.text.data)
# #     #     for i in range(0,100):
# #     #         for ii in range(batch.text.data.shape[0]):
# #     #             print(vocab.vocab.itos[batch.text.data[ii][i]],end="")
# #     #         print('')
# #     #     break
#
print(vocab.vocab.itos)
word2ix = vocab.vocab.stoi
ix2word = vocab.vocab.itos

np.savez('word2ix.npz', word2ix = word2ix,ix2word = ix2word)

# raw_to_train('data/raw_data.txt')
# split_data('data/train_data.txt')
# segmentation('data/train.txt')

