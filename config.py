class Config(object):
    data_path = 'data/' # 文本文件存放路径
    lr = 1e-3
    weight_decay = 1e-4
    use_gpu = True
    epoch = 20
    dropout = 0.0
    batch_size = 100
    max_gen_len = 50
    plot_every = 500  # 每200个batch 可视化一次
    # use_env = True  # 是否使用visodm
    env = 'charlstm'  # visdom env
    debug_file = '/tmp/debugp'
    model_path = None  # 预训练模型路径
    model_prefix = 'checkpoints/'  # 模型保存路径
    model = 'lstm_att_twin'
    pickle_path = 'data/word2ix.npz' #保存之前的word2ix
    logging_file = 'logging/lstm_att_twin.log' # 指定日志文件

opt = Config()