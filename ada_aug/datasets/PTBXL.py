import os
import pickle
from scipy import stats

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from .base import BaseDataset

DEFAULT_PATH = "/mnt/data2/teddy"
MAX_LENGTH = 1000
LABEL_GROUPS = {"all":71, "diagnostic":44, "subdiagnostic":23, "superdiagnostic":5, "form":19, "rhythm":12}

class PTBXL(BaseDataset):
    def __init__(self, dataset_path, labelgroup="diagnostic",multilabel=True,preprocess=[],transfroms=[],augmentations=[],label_transfroms=[],**_kwargs):
        super(PTBXL,self).__init__(preprocess=preprocess,transfroms=transfroms,augmentations=augmentations,label_transfroms=label_transfroms)
        assert labelgroup in ["all", "diagnostic", "subdiagnostic", "superdiagnostic", "form", "rhythm"]
        self.dataset_path = dataset_path
        self.max_len = MAX_LENGTH
        self.labelgroup = labelgroup
        self.num_class = LABEL_GROUPS[labelgroup]
        self.multilabel = multilabel
        self.channel = 12
        self._get_data()
    
    def _get_data(self,mode='all'):
        #file_list = os.listdir(os.path.join(self.dataset_path,self.labelgroup))
        self.input_data = None
        self.label = None
        if mode=='all':
            datas,labels = [0,0,0],[0,0,0]
            for i,type in enumerate(['train','val','test']):
                each_Xf = 'X_%s.npy'%type
                each_yf = 'y_%s.npy'%type
                datas[i] = np.load(os.path.join(self.dataset_path,self.labelgroup,each_Xf),allow_pickle=True)
                labels[i] = np.load(os.path.join(self.dataset_path,self.labelgroup,each_yf),allow_pickle=True)
            self.input_data = np.concatenate(datas,axis=0)
            self.label = np.concatenate(labels,axis=0)
        else:
            each_Xf = 'X_%s.npy'%mode
            each_yf = 'y_%s.npy'%mode
            self.input_data = np.load(os.path.join(self.dataset_path,self.labelgroup,each_Xf),allow_pickle=True)
            self.label = np.load(os.path.join(self.dataset_path,self.labelgroup,each_yf),allow_pickle=True)

        assert self.input_data != None
        assert self.label != None
        if not self.multilabel:
            label_count = np.sum(self.label,axis=1)
            single_label = (label_count==1)
            # no_label = (label_count==0) no normal in PTBXL
            self.input_data = self.input_data(single_label)
            self.label = self.label(single_label)
            self.label = torch.argmax(self.label, dim=1).reshape(-1) #back to int
