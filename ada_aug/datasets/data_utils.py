from typing import List
from random import sample
import random
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import pickle
import matplotlib.pyplot as plt

def preprocess_signals(X_train, X_validation, X_test, outputfolder):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:,np.newaxis].astype(float))
    
    # Save Standardizer data
    with open(outputfolder+'standard_scaler.pkl', 'wb') as ss_file:
        pickle.dump(ss, ss_file)

    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)

def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp

def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    stroke = [d[0] for d in data]
    target = torch.LongTensor([d[1] for d in data])
    stroke_length = [len(sq) for sq in stroke]
    stroke = rnn_utils.pad_sequence(stroke, batch_first=True, padding_value=0)
    return stroke, stroke_length, target

def plot_tseries(data,lead=-1):
    if lead>=0:
        plt.plot(data[:,lead])
    else:
        features_num = data.shape[1]
        for i in range(features_num):
            plt.plot(data[:,i])
    plt.show()

class Temporal_dataset(Dataset):

    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.target[index]


class Exemplar_dataset(Dataset):

    def __init__(self, data, stroke_length: List[int], idx: int, seed: int):
        self.data = data
        self.stroke_length = stroke_length
        self.idx = idx
        self.seed = seed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.stroke_length[index], self.idx

    def subsample(self, k: int):
        remove_cnt: int = self.__len__() - k
        if remove_cnt > 0:
            random.seed(self.seed)
            remove_idx = sample(range(self.__len__()), k=remove_cnt)
            for idx in sorted(remove_idx, reverse=True):
                del self.data[idx]
                del self.stroke_length[idx]


class Exemplar_teacher(Dataset):

    def __init__(self, data, stroke_length: List[int], idx: int, teacher, seed: int):
        self.data = data
        self.stroke_length = stroke_length
        self.idx = idx
        self.teacher = teacher
        self.seed = seed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.stroke_length[index], self.idx, self.teacher[index]

    def subsample(self, k: int):
        remove_cnt: int = self.__len__() - k
        if remove_cnt > 0:
            random.seed(self.seed)
            remove_idx = sample(range(self.__len__()), k=remove_cnt)
            for idx in sorted(remove_idx, reverse=True):
                del self.data[idx]
                del self.stroke_length[idx]
                del self.teacher[idx]
