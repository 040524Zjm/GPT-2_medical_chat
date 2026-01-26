# -*- coding: utf-8 -*-
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from parameter_config import *

import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
from .dataset import *
from parameter_config import *

params = ParameterConfig()

def load_dataset(train_path, valid_path):
    with open(train_path, 'rb') as f:
        train_input_list = pickle.load(f)

    with open(valid_path, 'rb') as f:
        valid_input_list = pickle.load(f)

    train_dataset = MyDataset(train_input_list, 300)
    valid_dataset = MyDataset(valid_input_list, 300)
    return train_dataset, valid_dataset

def collate_fn(batch):
    """
    将数据集的样本进行批处理
    :param batch:
    :return:
    """
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0) # 填充，长度一致
    label = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100) # -100不参与梯度
    return input_ids, label

def get_dataloader(train_path, valid_path):
    train_dataset, valid_dataset = load_dataset(train_path, valid_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True, # drop_last=True 表示最后一批数据不足batch_size时，不进行训练
    )

    validate_dataloader = DataLoader(
        valid_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    return train_dataloader, validate_dataloader


if __name__ == '__main__':
    train_path = '../data/medical_train.pkl'
    valid_path = '../data/medical_valid.pkl'
    # load_dataset(train_path
    train_dataloader, validate_dataloader = get_dataloader(train_path, valid_path)
    for input_ids, labels in train_dataloader:
        print('Hello World')
        print(f'input_ids-->{input_ids.shape}')
        print(f'labels-->{labels.shape}')
        print(f'labels-->{labels}')
        print(f'input_ids-->{input_ids}')

        print('*' * 80)
        break




