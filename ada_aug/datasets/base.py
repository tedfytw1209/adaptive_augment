import os
import pickle
from scipy import stats

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp

class BaseDataset(Dataset):
    def __init__(self,transfroms=[],augmentations=[],class_augmentations=[],label_transfroms=[],**_kwargs):
        #common args
        self.transfroms = transfroms
        self.augmentations = augmentations
        self.class_augmentations = class_augmentations
        self.label_transfroms = label_transfroms
        self.input_data = None
        self.label = None
        self.max_len = 100
        self.preprocessor = []

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, index):
        input_data, label = self.input_data[index], self.label[index]
        for transfrom in self.transfroms:
            input_data = transfrom(input_data)
        for augmentation in self.augmentations:
            input_data = augmentation(input_data)
        for augmentation in self.class_augmentations:
            input_data = augmentation(input_data,label)
        for label_trans in self.label_transfroms:
            label = label_trans(label)
        #input_data_tmp = torch.zeros(self.max_len, input_data.shape[1])
        #input_data_tmp = input_data[0:self.max_len]
        return input_data,len(input_data), label
    
    def _get_data(self):
        return
    
    def fit_preprocess(self,preprocessor, indexs=[]):
        if len(indexs)!=0:
            preprocessor.fit(np.vstack(self.input_data[indexs]).flatten()[:,np.newaxis].astype(float))
        else:
            preprocessor.fit(np.vstack(self.input_data).flatten()[:,np.newaxis].astype(float))
        self.preprocessor = [preprocessor]
        return preprocessor
    def trans_preprocess(self,preprocessor):
        self.input_data = apply_standardizer(self.input_data,preprocessor)
        return preprocessor
    def set_preprocess(self, preprocessor):
        self.preprocessor = [preprocessor]
        return