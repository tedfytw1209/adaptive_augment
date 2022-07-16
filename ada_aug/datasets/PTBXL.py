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

class PTBXL(BaseDataset):
    def __init__(self, dataset_path, labelgroup="diagnostic",multilabel=True,mode='all',preprocess=[],transfroms=[],augmentations=[],label_transfroms=[],**_kwargs):
        super(PTBXL,self).__init__(preprocess=preprocess,transfroms=transfroms,augmentations=augmentations,label_transfroms=label_transfroms)
        assert labelgroup in ["all", "diagnostic", "subdiagnostic", "superdiagnostic", "form", "rhythm"]
        self.dataset_path = dataset_path
        self.max_len = MAX_LENGTH
        self.labelgroup = labelgroup
        self.num_class = LABEL_GROUPS[labelgroup]
        self.multilabel = multilabel
        self.channel = 12
        if self.multilabel:
            print('Using multilabel')
        else:
            print('Using singlelabel')
        if mode!='all':
            print('Using default train/valid/test split: 8:1:1')
        if mode in ['val','valid']:
            mode = 'val'
        self._get_data(mode=mode)
    
    def _get_data(self,mode='all'):
        #file_list = os.listdir(os.path.join(self.dataset_path,self.labelgroup))
        self.input_data = None
        self.label = None
        X_from = 'X_%s.npy'
        if self.multilabel:
            y_from = 'y_%s.npy'
        else:
            y_from = 'y_%s_single.npy'
        if mode=='all':
            datas,labels = [0,0,0],[0,0,0]
            for i,type in enumerate(['train','val','test']):
                datas[i] = np.load(os.path.join(self.dataset_path,self.labelgroup,X_from%type),allow_pickle=True)
                labels[i] = np.load(os.path.join(self.dataset_path,self.labelgroup,y_from%type),allow_pickle=True)
            self.input_data = np.concatenate(datas,axis=0).astype(float)
            self.label = np.concatenate(labels,axis=0).astype(int)
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

        '''if not self.multilabel:
            label_count = np.sum(self.label,axis=1)
            single_label = (label_count==1)
            # no_label = (label_count==0) no normal in PTBXL
            self.input_data = self.input_data[single_label]
            self.label = self.label[single_label]
            self.label = torch.argmax(self.label, dim=1).reshape(-1) #back to int'''
