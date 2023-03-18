from cProfile import label
import os
import pickle
from scipy import stats,signal
from scipy.io import loadmat
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from .base import BaseDataset
from sklearn.model_selection import StratifiedKFold,KFold


MAX_LENGTH = 5000
rhythm_classes_cinc = ['SB', 'NSR', 'AF', 'STach', 'AFL', 'SI', 'SVT', 'ATach', 'AVNRT', 'SAAWR'] #SI(Sinus Irregularity), AVNRT, SAAWR can't find
rhythm_classes_ori = ['SB', 'SR', 'AFIB', 'ST', 'AF', 'SI', 'SVT', 'AT', 'AVNRT', 'SAAWR']
rhythm_sup = ['AFIB','GSVT','SB','SR']
rhythm_sup_dic = {'SB':'SB', 'SR':'SR', 'AFIB':'AFIB', 'ST':'GSVT', 'AF':'AFIB', 'SI':'SR', 'SVT':'GSVT', 'AT':'GSVT', 'AVNRT':'GSVT', 'SAAWR':'GSVT'}
rhythm_dic = {k:v for (k,v) in zip(rhythm_classes_cinc,rhythm_classes_ori)}
LABEL_GROUPS = {"yuall":12,"all":11,"all2":6, "rhythm":6, 'superrhythm':4}

class Chapman(BaseDataset):
    Hz = 500
    def __init__(self, dataset_path,labelgroup='all',mode='all',seed=42,multilabel=False,transfroms=[],augmentations=[],label_transfroms=[],**_kwargs):
        super(Chapman,self).__init__(transfroms=transfroms,augmentations=augmentations,label_transfroms=label_transfroms)
        self.dataset_path = dataset_path
        self.max_len = MAX_LENGTH
        self.num_class = LABEL_GROUPS[labelgroup]
        self.multilabel = multilabel
        if multilabel:
            self.lb = 'ml'
        else:
            self.lb = 'single'
        self.channel = 12
        self.labelgroup = labelgroup
        self.sub_tr_ratio = 1.0
        self.Hz = 500
        #k, ratio, sub_tr_idx, valid_idx, test_idx
        self.split_indices = []
        self.fold_indices = [] # fold_idx : indices
        if self.labelgroup in ['rhythm','superrhythm'] and self.multilabel:
            print('Rhythm only have single label')
            exit()
        if not self._check_data():
            self.process_data()
        self._get_data(mode=mode,seed=seed)
    
    def _check_data(self):
        return os.path.isfile(os.path.join(self.dataset_path,f'X_{self.labelgroup}data_{self.lb}.npy')) and \
                os.path.isfile(os.path.join(self.dataset_path,f'y_{self.labelgroup}data_{self.lb}.npy'))
    def _get_data(self,mode='all',seed=42):
        self.input_data = None
        self.label = None
        self.input_data = np.load(os.path.join(self.dataset_path,f'X_{self.labelgroup}data_{self.lb}.npy'),allow_pickle=True)
        self.label = np.load(os.path.join(self.dataset_path,f'y_{self.labelgroup}data_{self.lb}.npy'),allow_pickle=True)
        print('Label counts:')
        unique, counts = np.unique(self.label, return_counts=True)
        counts_array = np.asarray((unique, counts)).T
        print(counts_array)
        #Auto make self.split_indices / self.fold_indices
        all_indices = np.arange(len(self.label))
        n_fold = 10
        if not self.multilabel:
            skf = StratifiedKFold(n_splits=n_fold,random_state=seed, shuffle=True)
        else:
            skf = KFold(n_splits=n_fold,random_state=seed, shuffle=True)
        tot_len = 0
        for train_index, test_index in skf.split(self.input_data, self.label):
            self.fold_indices.append(test_index) #give fold indexs
            tot_len += len(test_index)
        assert tot_len==len(self.label)
        for test_k in range(n_fold):
            tr_idx,valid_idx,test_idx = np.array([]),None,None
            for i in range(n_fold):
                fold_idx = self.fold_indices[i]
                if i==test_k:
                    test_idx = fold_idx
                elif i==(test_k-1+10)%10:
                    valid_idx = fold_idx
                else:
                    tr_idx = np.concatenate([tr_idx,fold_idx],axis=0).astype(int)
            self.split_indices.append([test_k, self.sub_tr_ratio, tr_idx, valid_idx, test_idx])
        #make split indice
        if isinstance(mode,list):
            select_idxs = np.array([])
            for fold in mode: # fold:1~10, fold_indices:0~9
                select_idxs = np.concatenate([select_idxs,self.fold_indices[fold-1]],axis=0).astype(int)
            self.input_data = self.input_data[select_idxs]
            self.label = self.label[select_idxs]
        
    def down_sample(self,new_Hz):
        bs,seq_len,ch = self.input_data.shape
        second = int(seq_len/self.Hz)
        new_len = second * new_Hz
        #need (seq_len,ch)
        print('old data shape: ')
        print(self.input_data.shape)
        X_tmp = []
        for x in self.input_data:
            x_tmp = signal.resample(x,new_len)
            X_tmp.append(x_tmp)
        self.input_data = np.array(X_tmp)
        print('new data shape: ')
        print(self.input_data.shape)
        #
        self.Hz = new_Hz

    def process_data(self):
        print('Process data')
        #dx dic
        dx_df = pd.read_csv(os.path.join(self.dataset_path,'Dx_map.csv'))
        dx_dic = {str(k):str(v) for (k,v) in zip(dx_df['SNOMED CT Code'],dx_df['Abbreviation'])}
        print(dx_dic)
        select = None
        trans_dic = None
        outlabel = None
        filter_data = True
        occurance = 300
        if self.labelgroup=='rhythm':
            select = rhythm_classes_cinc
            trans_dic = None
            outlabel = rhythm_classes_ori
            cinc_sel = True
        elif self.labelgroup=='superrhythm':
            select = rhythm_classes_cinc
            trans_dic = rhythm_sup_dic
            outlabel = rhythm_sup
            cinc_sel = True
        elif self.labelgroup=='all':
            cinc_sel = True
        elif self.labelgroup=='all2':
            cinc_sel = True
            occurance = 600
        elif self.labelgroup=='yuall':
            cinc_sel = False
        elif self.labelgroup=='mlall':
            cinc_sel = False
            filter_data = False
            assert self.multilabel #only can use in multilabel
        else:
            print('label group error')
            exit()
        
        df = pd.DataFrame()
        labels = set()
        for i in range(1, 10647): # 10646
            try:
                with open(os.path.join(self.dataset_path,'raw','JS%05d.hea'%i), 'r') as f:
                    header = f.read()
                #print(header)
                #print(m['val'].shape)
            except:
                continue
            for l in header.split('\n'):
                if l.startswith('#Dx'):
                    entries = l.split(': ')[1].split(',')
                    abb_names = []
                    for entry in entries:
                        #print(entry)
                        if cinc_sel:
                            abb_name = dx_dic.get(entry,None)
                        else:
                            abb_name = str(entry)
                        if abb_name != None:
                            if select!=None and abb_name not in select:
                                continue
                            elif select!=None:
                                abb_name = rhythm_dic[abb_name]
                            if trans_dic:
                                abb_name = trans_dic[abb_name]
                            df.loc[i, abb_name] = 1
                            df.loc[i, 'id'] = i
                            df.loc[i, 'filename'] = 'WFDB_ChapmanShaoxing/raw/JS%05d'%i
                            labels.add(abb_name.strip())
                            abb_names.append(abb_name)
                        else:
                            print(entries)
                    #print(entry.strip(), end=' ')
                    df.loc[i, 'count'] = len(abb_names)
            #break
        df = df.dropna(axis=0,how='any',subset=['id'])
        df = df.fillna(0)
        print(labels)
        df.to_csv(f'raw{self.labelgroup}_{self.lb}.csv')
        fixed_col = ['id', 'filename', 'count', 'new_count']
        # del low number columns
        cols = []
        for col in df.columns:
            if col in fixed_col: continue
            yes_cnt = df[col].value_counts()[1]
            if filter_data and yes_cnt < occurance: 
                df = df.drop(columns=col)
            else:
                cols.append(col)
        #index reset
        df = df.reset_index(drop=True)
        df = df.drop(columns=['count','filename'])
        if outlabel==None:
            outlabel = sorted(cols)
        else:
            outlabel = [n for n in outlabel if n in cols] #only if >300
        df = df.reindex(columns=['id']+outlabel)
        df.to_csv(f'test{self.labelgroup}_{self.lb}.csv')
        #read data
        id_list = df['id'].values
        input_data = []
        for id in id_list:
            m = loadmat(os.path.join(self.dataset_path,'raw',"JS%05d.mat"%id))
            input_data.append(m['val'].T)
        input_data = np.array(input_data) #bs X L X channel
        labels = df.drop(columns='id').values
        #multilabel or singlelabel
        label_sum = np.sum(labels,axis=1)
        unique, counts = np.unique(label_sum, return_counts=True)
        counts_array = np.asarray((unique, counts)).T
        counts_data = pd.DataFrame(counts_array,columns=['label_counts','number'])
        counts_data.to_csv(f'label_counts{self.labelgroup}_{self.lb}.csv')
        if not self.multilabel:
            input_data = input_data[(label_sum==1)]
            labels = labels[(label_sum==1)]
            labels = np.argmax(labels, axis=1)
            final_df = df[(label_sum==1)]
        else:
            final_df = df
        final_df.to_csv(f'final_test{self.labelgroup}_{self.lb}.csv')
        print('Data shape: ',input_data.shape)
        print('Labels shape: ',labels.shape)
        np.save(os.path.join(self.dataset_path,f'X_{self.labelgroup}data_{self.lb}.npy'),input_data)
        np.save(os.path.join(self.dataset_path,f'y_{self.labelgroup}data_{self.lb}.npy'),labels)
        #self.input_data = input_data
        #self.label = labels
    def get_split_indices(self):
        return self.split_indices
