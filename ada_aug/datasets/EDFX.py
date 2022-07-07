import os
import pickle
from scipy import stats

import numpy as np
import torch
from torch.utils.data import Dataset,Subset
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

class EDFX(BaseDataset):
    def __init__(self, dataset_path,preprocess=[],transfroms=[],augmentations=[],label_transfroms=[],**_kwargs):
        super(EDFX,self).__init__(preprocess=preprocess,transfroms=transfroms,augmentations=augmentations,label_transfroms=label_transfroms)
        self.dataset_path = dataset_path
        self.max_len = MAX_LENGTH
        self.dataset = None
        self.multilabel = False
        self.channel = 2
        self.prep_physionet_dataset(mne_data_path=dataset_path,n_subj=81,recording_ids=[1],preload=True)
    
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
        '''print(len(windows_dataset))
        sample = windows_dataset[0]
        print(sample[0])
        print(sample[0].shape)
        print(sample[1])
        print(sample[2])'''
        self.dataset = windows_dataset
        #return windows_dataset, ['Fpz', 'Pz'], 100
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.dataset[index]
        input_data = sample[0].T
        label = sample[1]
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
        return input_data_tmp,len(input_data), label
