import os
import pickle
from scipy import stats

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from .base import BaseDataset
from biosppy.signals import tools

#DEFAULT_PATH = "/mnt/data2/teddy"
HZ, MAX_LEN = 100,60
MAX_LENGTH = HZ * MAX_LEN
LABEL_GROUPS = {"all":9}

def input_resizeing(raw_data,ratio=1):
    input_data = np.zeros((len(raw_data),int(MAX_LENGTH*ratio),12))
    seq_lens = np.zeros(len(raw_data))
    for idx, data in enumerate(raw_data):
        input_data[idx,:data.shape[0],:data.shape[1]] = tools.normalize(data[0:int(MAX_LENGTH*ratio),:])['signal']
        seq_lens[idx] = min(data.shape[0],int(MAX_LENGTH*ratio))
    return input_data, seq_lens

def make_split_indices(test_k,each_lens,sub_tr_ratio):
    tot_k = len(each_lens)
    tr_idx,valid_idx,test_idx = np.array([]),None,None
    start = 0
    for i in range(tot_k):
        each_len = each_lens[i]
        fold_idx = np.arange(each_len) + start
        if i==test_k:
            test_idx = fold_idx
        elif i==(test_k-1+10)%10:
            valid_idx = fold_idx
        else:
            tr_idx = np.concatenate([tr_idx,fold_idx],axis=0).astype(int)

        start += each_len
    #k, ratio, sub_tr_idx, valid_idx, test_idx
    return [test_k, sub_tr_ratio, tr_idx, valid_idx, test_idx]

class ICBEB(BaseDataset):
    Hz = 100
    def __init__(self, dataset_path, labelgroup="all",multilabel=True,mode='all',resize_ratio=1.0,
    transfroms=[],augmentations=[],class_augmentations=[],label_transfroms=[],**_kwargs):
        super(ICBEB,self).__init__(transfroms=transfroms,augmentations=augmentations,
            class_augmentations=class_augmentations,label_transfroms=label_transfroms)
        assert labelgroup in ["all"]
        self.dataset_path = dataset_path
        self.max_len = MAX_LENGTH
        self.labelgroup = labelgroup
        self.num_class = LABEL_GROUPS[labelgroup]
        self.multilabel = multilabel
        self.resize_ratio = resize_ratio
        self.channel = 12
        self.sub_tr_ratio = 1.0
        self.Hz = 100
        #k, ratio, sub_tr_idx, valid_idx, test_idx
        self.split_indices = [[0, self.sub_tr_ratio, 0, 0, 0]]
        if self.multilabel:
            print('Using multilabel')
        else:
            print('Using singlelabel')
        if isinstance(mode,list):
            print('Using 10-fold classification fold ',mode)
            self._get_data_fold(folds=mode)
        else:
            if mode!='all':
                print('Using default train/valid/test split: 8:1:1')
            if mode in ['val','valid']:
                mode = 'val'
            self._get_data(mode=mode)
    
    def _get_data(self,mode='all'):
        #file_list = os.listdir(os.path.join(self.dataset_path,self.labelgroup))
        self.input_data = None
        self.label = None
        if self.multilabel:
            X_from = 'X_%s_ml.npy'
        else:
            X_from = 'X_%s_single.npy'
        if self.multilabel:
            y_from = 'y_%s_ml.npy'
        else:
            y_from = 'y_%s_single.npy'
        if mode=='all':
            datas,labels = [0,0,0],[0,0,0]
            seq_lens = [0,0,0]
            start = 0
            for i,type in enumerate(['train','val','test']):
                datas[i] = np.load(os.path.join(self.dataset_path,self.labelgroup,X_from%type),allow_pickle=True)
                label = np.load(os.path.join(self.dataset_path,self.labelgroup,y_from%type),allow_pickle=True)
                datas[i],seq_lens[i] = input_resizeing(datas[i])
                labels[i] = label
                each_len = len(label)
                self.split_indices[0][i+2] = np.arange(each_len) + start
                start += each_len
            self.input_data = np.concatenate(datas,axis=0).astype(float)
            self.label = np.concatenate(labels,axis=0).astype(int)
            self.seq_lens = np.concatenate(seq_lens,axis=0).astype(int)
        elif mode=='foldall':
            X_from = 'X_fold%d_fix.npy'
            if self.multilabel:
                y_from = 'y_fold%d_ml.npy'
            else:
                y_from = 'y_fold%d_single.npy'
            datas,labels,seq_lens = [],[],[]
            each_lens = []
            for f in range(1,11):
                data = np.load(os.path.join(self.dataset_path,self.labelgroup,X_from%f),allow_pickle=True)
                data,seq_len = input_resizeing(data)
                label = np.load(os.path.join(self.dataset_path,self.labelgroup,y_from%f),allow_pickle=True)
                datas.append(data)
                labels.append(label)
                each_len = len(label)
                each_lens.append(each_len)
                seq_lens.append(seq_len)
            self.input_data = np.concatenate(datas,axis=0).astype(float)
            self.label = np.concatenate(labels,axis=0).astype(int)
            self.seq_lens = np.concatenate(seq_lens,axis=0).astype(int)
            self.split_indices = [] #k, ratio, sub_tr_idx, valid_idx, test_idx
            for k in range(10): #10folds
                each_split = make_split_indices(k,each_lens,self.sub_tr_ratio)
                self.split_indices.append(each_split)
        elif mode=='tottrain':
            datas,labels = [0,0],[0,0]
            seq_lens = [0,0]
            for i,type in enumerate(['train','test']):
                datas[i] = np.load(os.path.join(self.dataset_path,self.labelgroup,X_from%type),allow_pickle=True)
                datas[i],seq_lens[i] = input_resizeing(datas[i])
                labels[i] = np.load(os.path.join(self.dataset_path,self.labelgroup,y_from%type),allow_pickle=True)
            self.input_data = np.concatenate(datas,axis=0).astype(float)
            self.label = np.concatenate(labels,axis=0).astype(int)
            self.seq_lens = np.concatenate(seq_lens,axis=0).astype(int)
        else:
            self.input_data = np.load(os.path.join(self.dataset_path,self.labelgroup,X_from%mode),allow_pickle=True).astype(float)
            self.input_data, self.seq_lens = input_resizeing(self.input_data)
            self.label = np.load(os.path.join(self.dataset_path,self.labelgroup,y_from%mode),allow_pickle=True).astype(int)

    def _get_data_fold(self,folds):
        self.input_data = None
        self.label = None
        if self.multilabel:
            y_from = 'y_fold%d_ml.npy'
            X_from = 'X_fold%d_ml.npy'
        else:
            y_from = 'y_fold%d_single.npy'
            X_from = 'X_fold%d_fix.npy'
        datas,labels,seq_lens = [],[],[]
        for f in folds:
            data = np.load(os.path.join(self.dataset_path,self.labelgroup,X_from%f),allow_pickle=True)
            data,seq_len = input_resizeing(data)
            label = np.load(os.path.join(self.dataset_path,self.labelgroup,y_from%f),allow_pickle=True)
            datas.append(data)
            labels.append(label)
            seq_lens.append(seq_len)
        self.input_data = np.concatenate(datas,axis=0).astype(float)
        self.label = np.concatenate(labels,axis=0).astype(int)
        self.seq_lens = np.concatenate(seq_lens,axis=0).astype(int)

    def get_split_indices(self):
        return self.split_indices
    #for dyn lens
    def __getitem__(self, index):
        input_data, label = self.input_data[index], self.label[index]
        seq_len = self.seq_lens[index]
        for transfrom in self.transfroms:
            input_data = transfrom(input_data,seq_len=seq_len)
        for augmentation in self.augmentations:
            input_data = augmentation(input_data,seq_len=seq_len)
        for augmentation in self.class_augmentations:
            input_data = augmentation(input_data,label,seq_len=seq_len)
        for label_trans in self.label_transfroms:
            label = label_trans(label)
        return input_data,seq_len, label
    #ICBEB special, no need for preprocess
    def fit_preprocess(self,preprocessor, indexs=[]):
        '''
        if len(indexs)==0:
            preprocessor.fit(np.vstack(self.input_data[indexs]).flatten()[:,np.newaxis].astype(float))
        else:
            preprocessor.fit(np.vstack(self.input_data).flatten()[:,np.newaxis].astype(float))
        '''
        return preprocessor
    def trans_preprocess(self,preprocessor):
        '''
        self.input_data = apply_standardizer(self.input_data,preprocessor)
        '''
        return preprocessor
    def set_preprocess(self, preprocessor):
        '''
        self.preprocessor = [preprocessor]
        '''
        self.preprocessor = []
        return