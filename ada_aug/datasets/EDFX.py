import os
import pickle
from scipy import stats
from collections.abc import Iterable
import numpy as np
import torch
from torch.utils.data import Dataset,Subset
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from .base import BaseDataset
import pandas as pd
from mne import set_log_level
from braindecode.datasets.sleep_physionet import SleepPhysionet
from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datautil.preprocess import zscore, MNEPreproc, NumpyPreproc,preprocess

MAX_LENGTH = 3000
TARGETS_MAPPING = {  # We merge stages 3 and 4 following AASM standards.
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4
}

def check_grid(grid, max_value=None, n_values=None):
    if isinstance(grid, Iterable):
        grid_values = list(grid)
    elif isinstance(grid, float) or grid is None:
        grid_values = [grid]
    else:
        raise ValueError(
            "grid can be either an iterable or a str.",
            f"Got {type(grid)}."
        )
    return grid_values
def get_groups(dataset):
    if (
        hasattr(dataset, "description") and
        hasattr(dataset, "datasets") and
        "subject" in dataset.description
    ):
        return np.hstack([
            [subj] * len(dataset.datasets[rec])
            for rec, subj in enumerate(
                dataset.description['subject'].values
            )
        ])
    else:
        return np.arange(len(dataset))

def _structured_split(
    splitter_class,
    indices,
    ratio,
    groups,
    targets=None,
    random_state=None
):
    if ratio == 1:
        return indices, np.array([])

    if isinstance(ratio, float):
        assert (
            ratio > 0 and ratio < 1
        ), "When ratio is a float, it must be positive and <=1."
    else:
        assert isinstance(ratio, int), (
            f"ratio can be either int or float. Got {type(ratio)}: {ratio}"
        )
    splitter = splitter_class(
        n_splits=1,
        train_size=ratio,
        random_state=random_state
    )
    train_idx, test_idx = list(splitter.split(
        indices,
        y=targets,
        groups=groups
    ))[0]
    return indices[train_idx], indices[test_idx]


def grouped_split(indices, ratio, groups, random_state=None):
    return _structured_split(
        splitter_class=GroupShuffleSplit,
        indices=indices,
        ratio=ratio,
        groups=groups,
        random_state=random_state
    )
def stratified_split(indices, ratio, targets, random_state=None):
    return _structured_split(
        splitter_class=StratifiedShuffleSplit,
        indices=indices,
        ratio=ratio,
        groups=None,
        targets=targets,
        random_state=random_state
    )

def _get_split_indices(
    cv,
    windows_dataset,
    groups,
    train_size_over_valid,
    data_ratios,
    max_ratios,
    grouped_subset=True,
    random_state=None
):
    
    if data_ratios is None:
        data_ratios = [1.]
    if not grouped_subset:
        targets = np.array([y for _, y, _ in windows_dataset])

    splits_proportions = list()
    fold_indices = []
    for k, fold in enumerate(
        cv.split(windows_dataset, groups=groups),
        start=1
    ):
        train_and_valid_idx, test_idx = fold

        train_idx, valid_idx = grouped_split(
            indices=train_and_valid_idx,
            ratio=train_size_over_valid,
            groups=groups[train_and_valid_idx],
            random_state=random_state
        )
        for ratio in check_grid(data_ratios, len(train_idx), max_ratios):
            if grouped_subset:
                sub_tr_idx, _ = grouped_split(
                    indices=train_idx,
                    ratio=ratio,
                    groups=groups[train_idx],
                    random_state=random_state
                )
            else:
                sub_tr_idx, _ = stratified_split(
                    indices=train_idx,
                    ratio=ratio,
                    targets=targets[train_idx],
                    random_state=random_state
                )
            splits_proportions.append(
                (k, ratio, sub_tr_idx, valid_idx, test_idx)
            )
            fold_indices.append(test_idx) #index of each fold
    return splits_proportions, fold_indices

class EDFX(BaseDataset):
    Hz = 100
    def __init__(self, dataset_path,mode='all',transfroms=[],augmentations=[],label_transfroms=[],n_folds=5,seed=42,**_kwargs):
        super(EDFX,self).__init__(transfroms=transfroms,augmentations=augmentations,label_transfroms=label_transfroms)
        self.dataset_path = dataset_path
        self.max_len = MAX_LENGTH
        self.dataset = None
        self.multilabel = False
        self.channel = 2
        self.sub_tr_ratio = 1.0
        self.num_class = 5
        self.max_len = 3000
        self.n_folds = n_folds
        self.Hz = 100
        if not self._check_data():
            self.prep_physionet_dataset(mne_data_path=dataset_path,n_subj=81,recording_ids=[1],preload=True)
        self._get_data(mode=mode,seed=seed)
    
    def _check_data(self):
        return os.path.isfile(os.path.join(self.dataset_path,f'X_data.npy')) and \
                os.path.isfile(os.path.join(self.dataset_path,f'y_data.npy'))
    
    def _get_data(self,mode='all',seed=42):
        self.input_data = None
        self.label = None
        self.input_data = np.load(os.path.join(self.dataset_path,f'X_{self.labelgroup}data_{self.lb}.npy'),allow_pickle=True)
        self.label = np.load(os.path.join(self.dataset_path,f'y_{self.labelgroup}data_{self.lb}.npy'),allow_pickle=True)
        print('Label counts:')
        unique, counts = np.unique(self.label, return_counts=True)
        counts_array = np.asarray((unique, counts)).T
        print(counts_array)
        if mode=='all':
            print("Using origin data format")
        elif isinstance(mode,list): #make split indice
            train_ovr = (self.n_folds - 2)
            splits_proportions,fold_indices = self.CV_split_indices(train_ovr,random_state=seed)
            self.split_indices = splits_proportions
            self.fold_indices = fold_indices
            select_idxs = np.array([])
            for fold in mode: # fold:1~10, fold_indices:0~9
                select_idxs = np.concatenate([select_idxs,self.fold_indices[fold-1]],axis=0).astype(int)
            self.input_data = self.input_data[select_idxs]
            self.label = self.label[select_idxs]
            
    
    def prep_physionet_dataset(
        self,
        mne_data_path=None, #
        n_subj=None, #81
        recording_ids=None, #[1]
        window_size_s=30,
        sfreq=100,
        should_preprocess=True,
        should_normalize=True,
        high_cut_hz=30,
        crop_wake_mins=30,
        crop=None,
        preload=False, #True
        ):
        """Import, create and preprocess SleepPhysionet dataset.

        Parameters
        ----------
        mne_data_path : str, optional
        Path to put the fetched data in. By default None
        n_subj : int | None, optional
        Number of subjects to import. If omitted, all subjects will be imported
        and used.
        recording_ids : list | None, optional
        List of recoding indices (int) to be imported per subject. If ommited,
        all recordings will be imported and used (i.e. [1,2]).
        window_size_s : int, optional
        Window size in seconds defining each sample. By default 30.
        sfreq : int, optional
        Sampling frequency in Hz. by default 100
        should_preprocess : bool, optional
        Whether to preprocess the data with a low-pass filter and microvolts
        scaling. By default True.
        should_normalize : bool, optional
        Whether to normalize (zscore) the windows. By default True.
        high_cut_hz : int, optional
        Cut frequency to use for low-pass filter in case of preprocessing. By
        default 30.
        crop_wake_mins : int, optional
        Number of minutes of wake time to keep before the first sleep event
        and after the last sleep event. Used to reduce the imbalance in this
        dataset. Default of 30 mins.
        crop : tuple | None
        If not None, crop the raw data with (tmin, tmax). Useful for
        testing fast.
        preload : bool, optional
        Whether to preload raw signals in the RAM.
        Returns
        -------
        braindecode.datasets.BaseConcatDataset
        """

        if n_subj is None:
            subject_ids = None
        else:
            subject_ids = range(n_subj)
        set_log_level(False)
        dataset = SleepPhysionet(
            subject_ids=subject_ids,
            recording_ids=recording_ids,
            crop_wake_mins=crop_wake_mins,
            path=mne_data_path
        )
        preprocessors = [
            # convert from volt to microvolt, directly modifying the array
            NumpyPreproc(fn=lambda x: x * 1e6),
            # bandpass filter
            MNEPreproc(
            fn='filter',
            l_freq=None,
            h_freq=high_cut_hz,
            verbose=False
            ),
        ]

        if crop is not None:
            preprocessors.insert(
                1,
                MNEPreproc(
                fn='crop',
                tmin=crop[0],
                tmax=crop[1]
                )
            )

        if should_preprocess:
            # Transform the data
            preprocess(dataset, preprocessors, bar=True)

        window_size_samples = window_size_s * sfreq
        windows_dataset = create_windows_from_events(
            dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
            window_size_samples=window_size_samples,
            window_stride_samples=window_size_samples, preload=preload,
            mapping=TARGETS_MAPPING, verbose=False,
        )
        if should_normalize:
            preprocess(windows_dataset, [MNEPreproc(fn=zscore)])
        print(len(windows_dataset))
        sample = windows_dataset[0]
        print(sample[0])
        print(sample[0].shape)
        print(sample[1])
        print(sample[2])
        #self.dataset = windows_dataset
        input_data = []
        labels = []
        for sample in windows_dataset:
            input_data.append(sample[0])
            labels.append(sample[1])
        input_data = np.array(input_data)
        labels = np.array(labels)
        np.save(os.path.join(self.dataset_path,f'X_data.npy'),input_data)
        np.save(os.path.join(self.dataset_path,f'y_data.npy'),labels)
        #return windows_dataset
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        #sample = self.dataset[index] # output: (input,label,window timestep)
        input_data = self.input_data[index].T
        label = self.label[index]
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
        return input_data_tmp,len(input_data), label #(data,seq_len,label)

    def CV_split_indices(self, train_size_over_valid=0.5, data_ratios=None, max_ratios=None, grouped_subset=True, random_state=29):
        kf = GroupKFold(n_splits=self.n_folds)
        groups = get_groups(self.input_data)
        splits_proportions,fold_indices = _get_split_indices(cv=kf,
                windows_dataset=self.input_data,
                groups=groups,
                train_size_over_valid=train_size_over_valid,
                data_ratios=data_ratios,
                max_ratios=max_ratios,
                grouped_subset=grouped_subset,
                random_state=random_state)
        return splits_proportions,fold_indices #(k, ratio, sub_tr_idx, valid_idx, test_idx) * 5
