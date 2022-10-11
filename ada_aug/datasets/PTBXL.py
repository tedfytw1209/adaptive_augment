import os
import pickle
from scipy import stats

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from .base import BaseDataset

#DEFAULT_PATH = "/mnt/data2/teddy"
MAX_LENGTH = 1000
LABEL_GROUPS = {"all":71, "diagnostic":44, "subdiagnostic":23, "superdiagnostic":5, "form":19, "rhythm":12}

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

class PTBXL(BaseDataset):
    Hz = 100
    def __init__(self, dataset_path, labelgroup="diagnostic",multilabel=True,mode='all',denoise=False,
    transfroms=[],augmentations=[],class_augmentations=[],label_transfroms=[],**_kwargs):
        super(PTBXL,self).__init__(transfroms=transfroms,augmentations=augmentations,
            class_augmentations=class_augmentations,label_transfroms=label_transfroms)
        assert labelgroup in ["all", "diagnostic", "subdiagnostic", "superdiagnostic", "form", "rhythm"]
        self.dataset_path = dataset_path
        self.max_len = MAX_LENGTH
        self.labelgroup = labelgroup
        self.num_class = LABEL_GROUPS[labelgroup]
        self.multilabel = multilabel
        self.denoise = denoise
        self.channel = 12
        self.sub_tr_ratio = 1.0
        self.Hz = 100
        #k, ratio, sub_tr_idx, valid_idx, test_idx
        self.split_indices = [[0, self.sub_tr_ratio, 0, 0, 0]]
        if self.multilabel:
            print('Using multilabel')
        else:
            print('Using singlelabel')
        if self.denoise:
            print('Using denoise dataset')
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
            start = 0
            for i,type in enumerate(['train','val','test']):
                datas[i] = np.load(os.path.join(self.dataset_path,self.labelgroup,X_from%type),allow_pickle=True)
                label = np.load(os.path.join(self.dataset_path,self.labelgroup,y_from%type),allow_pickle=True)
                labels[i] = label
                each_len = len(label)
                self.split_indices[0][i+2] = np.arange(each_len) + start
                start += each_len
            self.input_data = np.concatenate(datas,axis=0).astype(float)
            self.label = np.concatenate(labels,axis=0).astype(int)
        elif mode=='foldall':
            X_from = 'X_fold%d_raw.npy'
            if self.multilabel:
                y_from = 'y_fold%d_ml.npy'
            else:
                y_from = 'y_fold%d_single.npy'
            datas,labels = [],[]
            each_lens = []
            for f in range(1,11):
                data = np.load(os.path.join(self.dataset_path,self.labelgroup,X_from%f),allow_pickle=True)
                label = np.load(os.path.join(self.dataset_path,self.labelgroup,y_from%f),allow_pickle=True)
                datas.append(data)
                labels.append(label)
                each_len = len(label)
                each_lens.append(each_len)
            self.input_data = np.concatenate(datas,axis=0).astype(float)
            self.label = np.concatenate(labels,axis=0).astype(int)
            self.split_indices = [] #k, ratio, sub_tr_idx, valid_idx, test_idx
            for k in range(10): #10folds
                each_split = make_split_indices(k,each_lens,self.sub_tr_ratio)
                self.split_indices.append(each_split)
        elif mode=='tottrain':
            datas,labels = [0,0],[0,0]
            for i,type in enumerate(['train','test']):
                datas[i] = np.load(os.path.join(self.dataset_path,self.labelgroup,X_from%type),allow_pickle=True)
                labels[i] = np.load(os.path.join(self.dataset_path,self.labelgroup,y_from%type),allow_pickle=True)
            self.input_data = np.concatenate(datas,axis=0).astype(float)
            self.label = np.concatenate(labels,axis=0).astype(int)
        else:
            self.input_data = np.load(os.path.join(self.dataset_path,self.labelgroup,X_from%mode),allow_pickle=True).astype(float)
            self.label = np.load(os.path.join(self.dataset_path,self.labelgroup,y_from%mode),allow_pickle=True).astype(int)

    def _get_data_fold(self,folds):
        self.input_data = None
        self.label = None
        if self.multilabel:
            y_from = 'y_fold%d_ml.npy'
            X_from = 'X_fold%d_ml.npy'
        else:
            y_from = 'y_fold%d_single.npy'
            X_from = 'X_fold%d_raw.npy'
        datas,labels = [],[]
        for f in folds:
            data = np.load(os.path.join(self.dataset_path,self.labelgroup,X_from%f),allow_pickle=True)
            label = np.load(os.path.join(self.dataset_path,self.labelgroup,y_from%f),allow_pickle=True)
            datas.append(data)
            labels.append(label)
        self.input_data = np.concatenate(datas,axis=0).astype(float)
        self.label = np.concatenate(labels,axis=0).astype(int)

    def get_split_indices(self):
        return self.split_indices