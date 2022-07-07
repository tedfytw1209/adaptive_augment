import os
import pickle
from scipy import stats
from scipy.io import loadmat
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from .base import BaseDataset


MAX_LENGTH = 5000

class Chapman(BaseDataset):
    def __init__(self, dataset_path,preprocess=[],transfroms=[],augmentations=[],label_transfroms=[],**_kwargs):
        super(Chapman,self).__init__(preprocess=preprocess,transfroms=transfroms,augmentations=augmentations,label_transfroms=label_transfroms)
        self.dataset_path = dataset_path
        self.max_len = MAX_LENGTH
        self.num_class = 12
        self.multilabel = True
        self.channel = 12
        if not self._check_data():
            self.process_data()
        self._get_data()
    
    def _check_data(self):
        return os.path.isfile(os.path.join(self.dataset_path,'X_data.npy')) and os.path.isfile(os.path.join(self.dataset_path,'y_data.npy'))
    def _get_data(self):
        self.input_data = None
        self.label = None
        self.input_data = np.load(os.path.join(self.dataset_path,'X_data.npy'),allow_pickle=True)
        self.label = np.load(os.path.join(self.dataset_path,'y_data.npy'),allow_pickle=True)

    def process_data(self):
        print('Process data')
        df = pd.DataFrame()
        labels = set()
        for i in range(1, 10647): # 10646
            try:
                with open(os.path.join(self.dataset_path,'JS%05d.hea'%i), 'r') as f:
                    header = f.read()
                #print(header)
                #print(m['val'].shape)
            except:
                continue
            for l in header.split('\n'):
                if l.startswith('#Dx'):
                    entries = l.split(': ')[1].split(',')
                    for entry in entries:
                        #print(entry)
                        df.loc[i, entry] = 1
                        df.loc[i, 'id'] = i
                        df.loc[i, 'filename'] = 'WFDB_ChapmanShaoxing/JS%05d'%i
                        labels.add(entry.strip())
                    #print(entry.strip(), end=' ')
                    df.loc[i, 'count'] = len(entries)
            #break
        df = df.fillna(0)
        fixed_col = ['id', 'filename', 'count', 'new_count']
        # 刪除數量少的columns
        for col in df.columns:
            if col in fixed_col: continue
            yes_cnt = df[col].value_counts()[1]
            if yes_cnt < 300: 
                df = df.drop(columns=col)
        df = df.reset_index(drop=True)
        cnt = 0
        new_col = []
        # 修改col名稱
        for col in df.columns:
            if col in fixed_col: 
                new_col.append(col)
            else:
                cnt += 1
                new_col.append('label%d'%cnt)

        df.columns = new_col
        df = df.drop(columns=['count','filename'])
        new_col.remove('id')
        new_col.remove('filename')
        new_col.remove('count')
        df = df.reindex(columns=['id']+new_col)
        #df.to_csv('test.csv')
        #read data
        id_list = df['id'].values
        input_data = []
        for id in id_list:
            m = loadmat(os.path.join(self.dataset_path,"JS%05d.mat"%id))
            input_data.append(m['val'].T)
        input_data = np.array(input_data) #bs X L X channel
        labels = df.drop(columns='id').values
        np.save(os.path.join(self.dataset_path,'X_data.npy'),input_data)
        np.save(os.path.join(self.dataset_path,'y_data.npy'),labels)
        #self.input_data = input_data
        #self.label = labels

