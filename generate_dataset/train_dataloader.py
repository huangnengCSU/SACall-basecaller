#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Filename: dataset
# @Date    : 02/19/2019
# @Author  : Neng Huang
# @Email   : csuhuangneng@gmail.com

import os
import torch
import torch.utils.data as Data
import numpy as np
import random
from tqdm import tqdm


class TrainBatchBasecallDataset(Data.Dataset):
    def __init__(self, signal_dir, label_dir):
        file_list = os.listdir(signal_dir)
        self.file_count = len(file_list)
        print('file number:', self.file_count)

        self.file_list = file_list
        self.signal_path = signal_dir
        self.label_path = label_dir
        self.signals = list()
        self.labels = list()
        self.idx = 0

    def __len__(self):
        return self.file_count

    def __getitem__(self, item):
        signal = np.load(self.signal_path+'/'+self.file_list[item])

        # label file has the same file name as signal file, but in different directory.
        label = np.load(self.label_path+'/'+self.file_list[item])
        return signal, label


class TrainBatchProvider():
    def __init__(self, dataset, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.dataiter = None
        self.sample_number = 0
        self.signal_pool, self.label_pool = [], []

    def build(self):
        dataloader = Data.DataLoader(
            self.dataset, batch_size=1, shuffle=self.shuffle)
        self.dataiter = dataloader.__iter__()

    def next(self):
        if self.dataiter is None:
            self.build()
        try:
            while self.sample_number < self.batch_size:
                signal, label = self.dataiter.next()
                signal = torch.squeeze(signal, dim=0)
                label = torch.squeeze(label, dim=0)
                self.sample_number += signal.shape[0]
                self.signal_pool.append(signal)
                self.label_pool.append(label)
            whole_signal = torch.cat(self.signal_pool, dim=0)
            whole_label = torch.cat(self.label_pool, dim=0)
            batch_signal = whole_signal[:self.batch_size]
            batch_label = whole_label[:self.batch_size]
            self.signal_pool = [whole_signal[self.batch_size:]]
            self.label_pool = [whole_label[self.batch_size:]]
            self.sample_number -= self.batch_size
            return torch.unsqueeze(batch_signal, dim=2), batch_label
        except StopIteration:
            return None, None
