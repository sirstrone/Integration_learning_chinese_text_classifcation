# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from utils import build_dataset, build_iterator, get_time_dif
from Model import Config,Integrated

def init_model(dataset, embedding, Config, Model):
    config = Config(dataset, embedding)
    model = Model(config).to(config.device)

    return config, model

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_WeiboChar.npz'
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    config,model = init_model(dataset,embedding,Config,Integrated)



    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config,None)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    print(model.parameters)
    train(config,model,train_iter, dev_iter, test_iter)
