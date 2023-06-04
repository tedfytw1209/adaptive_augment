import os
import pickle
from scipy import stats

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from .base import BaseDataset
from sklearn.model_selection import StratifiedKFold,KFold

BASE_PATH = "/mnt/data2/teddy"
MAX_LENGTH = 200

activity_code = {'A': 'walking', 'B': 'jogging', 'C': 'stairs', 'D': 'sitting', 'E': 'standing',
                 'F': 'typing', 'G': 'teeth', 'H': 'soup', 'I': 'chips', 'J': 'pasta',
                 'K': 'drinking', 'L': 'sandwich', 'M': 'kicking', 'O': 'catch', 'P': 'dribbling',
                 'Q': 'writing', 'R': 'clapping', 'S': 'folding'}
LABEL_GROUPS = {"all":18}

class_name = [
    i for i in range(18)
]


activity_label = {
    k: i for i, k in enumerate(activity_code)
}


subject_id = [
    str(i) for i in range(1600, 1651)
]


class WISDM(BaseDataset):
    Hz = 20
    def __init__(self,data_dir,labelgroup='all',mode='all',seed=42, sensor="accel", device="phone",**_kwargs):
        super(WISDM,self).__init__(**_kwargs)
        assert sensor in ["accel", "gyro"]
        assert device in ["phone", "watch"]
        self.data_dir = data_dir
        self.max_len = MAX_LENGTH
        self.num_class = LABEL_GROUPS[labelgroup]
        self.multilabel = False
        self.channel = 3
        self.labelgroup = labelgroup
        self.sub_tr_ratio = 1.0
        self.Hz = 20
        self.split_indices = []
        self.fold_indices = [] # fold_idx : indices
        self.loc = os.path.join(data_dir,"raw/", device, sensor)
        if not self.checkProcessed():
            self.process(sensor, device)
        #get data
        if mode=='all':
            input_datas , labels = [], []
            data_modes = ['train','valid','test']
            for i in range(3):
                input_data,label = self._get_data(data_modes[i])
                input_datas.extend(input_data)
                labels.extend(label)
            self.input_data = input_datas
            self.label = labels
        elif isinstance(mode,list) or mode=='foldall':
            input_datas , labels = [], []
            data_modes = ['train','valid','test']
            for i in range(3):
                input_data,label = self._get_data(data_modes[i])
                input_datas.extend(input_data)
                labels.extend(label)
            print('data len: ',len(input_datas))
            print('data[0] shape: ',input_datas[0].shape)
            input_datas = np.stack(input_datas)
            labels = np.array(labels).astype(int)
            print('input_datas shape: ',input_datas.shape)
            print('labels shape: ',labels.shape)
            #make fold
            all_indices = np.arange(len(labels))
            n_fold = 10
            if not self.multilabel:
                skf = StratifiedKFold(n_splits=n_fold,random_state=seed, shuffle=True)
            else:
                skf = KFold(n_splits=n_fold,random_state=seed, shuffle=True)
            tot_len = 0
            for lb,index in skf.split(all_indices,labels):
                self.fold_indices.append(index) #give fold indexs
                tot_len += len(index)
            assert tot_len==len(labels)
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
            #select fold
            if isinstance(mode,list):
                select_idxs = np.array([])
                for fold in mode: # fold:1~10, fold_indices:0~9
                    select_idxs = np.concatenate([select_idxs,self.fold_indices[fold-1]],axis=0).astype(int)
                self.input_data = input_datas[select_idxs]
                self.label = labels[select_idxs]
            else:
                self.input_data = input_datas
                self.label = labels
        elif mode=='tottrain':
            input_datas , labels = [], []
            data_modes = ['train','valid']
            for i in range(2):
                input_data,label = self._get_data(data_modes[i])
                input_datas.extend(input_data)
                labels.extend(label)
            self.input_data = input_datas
            self.label = labels
        else:
            input_datas,labels = self._get_data(mode=mode)
            self.input_data = input_datas
            self.label = labels
        print('Data and Label len:')
        print(len(self.input_data))
        print(len(self.label))
        self.num_class = len(class_name)
        self.channel = 3
        self.max_len = MAX_LENGTH
        self.Hz = 20

    def __len__(self):
        return len(self.input_data)
    # def __getitem__(self, index): #getitem use BaseDataset
    def checkProcessed(self):
        return os.path.isdir(self.loc.replace("raw", "processed"))

    def process(self, sensor, device,test_size=0.15,val_size=0.17):
        print("Process dataset")
        subject = subject_id
        labels = []
        input_datas = []
        for sub in subject:
            file_path = os.path.join(self.loc, f"data_{sub}_{sensor}_{device}.txt")
            with open(file_path) as f:
                file_data = [line.rstrip(";\n").split(",") for line in f.readlines()]
            x_axis = np.array([float(line[3]) for line in file_data])
            y_axis = np.array([float(line[4]) for line in file_data])
            z_axis = np.array([float(line[5]) for line in file_data])
            x_axis = (x_axis - np.mean(x_axis, axis=0)) / np.std(x_axis, axis=0)
            y_axis = (y_axis - np.mean(y_axis, axis=0)) / np.std(y_axis, axis=0)
            z_axis = (z_axis - np.mean(z_axis, axis=0)) / np.std(z_axis, axis=0)
            cnt = 0

            while cnt + MAX_LENGTH < len(file_data):
                input_data =\
                    torch.as_tensor([[x_axis[i], y_axis[i], z_axis[i]]
                                     for i in range(cnt, cnt + MAX_LENGTH)])
                label = stats.mode([file_data[i][1] for i in range(cnt, cnt + MAX_LENGTH)])
                labels.append(activity_label[label[0][0]])
                input_datas.append(input_data)
                cnt += MAX_LENGTH

        '''# Train test split
        X_train, X_test, y_train, y_test = train_test_split(input_datas, labels,
                                                            test_size=test_size, stratify=labels, random_state=42)
        # Train valid split
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                              test_size=val_size, stratify=y_train, random_state=42)'''

        if not os.path.isdir(os.path.join(self.data_dir,"/processed")):
            os.mkdir(os.path.join(self.data_dir,"/processed"))
        if not os.path.isdir(os.path.join(self.data_dir,f"/processed/{device}")):
            os.mkdir(os.path.join(self.data_dir,f"/processed/{device}"))
        if not os.path.isdir(os.path.join(self.data_dir,f"/processed/{device}/{sensor}")):
            os.mkdir(os.path.join(self.data_dir,f"/processed/{device}/{sensor}"))

        pickle.dump(labels, open(self.loc.replace("raw", "processed") + "/data_label", "wb"))
        pickle.dump(input_datas, open(self.loc.replace("raw", "processed") + "/data_input", "wb"))
    def _get_data(self,mode):
        input_datas = pickle.load(open(self.loc.replace("raw", "processed") + f"/{mode}_input", "rb"))
        labels = pickle.load(open(self.loc.replace("raw", "processed") + f"/{mode}_label", "rb"))
        return input_datas,labels
    def get_split_indices(self):
        return self.split_indices


if __name__ == "__main__":
    print(activity_label)
    print(class_name)
    wisdm = WISDM('/mnt/data2/teddy/modals-main/modals/datasets/wisdm-dataset')
    print(len(wisdm))
    print(wisdm[0][0].shape, wisdm[0][1])
