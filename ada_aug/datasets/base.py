import os
import pickle
from scipy import stats

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self,preprocess=[],transfroms=[],augmentations=[],label_transfroms=[],**_kwargs):
        #common args
        self.preprocess = preprocess
        self.transfroms = transfroms
        self.augmentations = augmentations
        self.label_transfroms = label_transfroms
        self.input_data = None
        self.label = None
        self.max_len = 100

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, index):
        input_data, label = self.input_data[index], self.label[index]
        for process in self.preprocess:
            input_data = process(input_data)
        for transfrom in self.transfroms:
            input_data = transfrom(input_data)
        for augmentation in self.augmentations:
            input_data = augmentation(input_data)
        for label_trans in self.label_transfroms:
            label = label_trans(label)
        input_data_tmp = torch.zeros(self.max_len, input_data.shape[1])
        input_data_tmp = input_data[0:self.max_len]
        return input_data_tmp,len(input_data), label
    
    def _get_data(self):
        return