from audioop import reverse
from enum import auto
from numbers import Real
from operator import invert
import random
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import Rbf
from scipy.spatial.transform import Rotation
from sklearn.utils import check_random_state
from torch.fft import fft, ifft
from torch.nn.functional import dropout2d, pad, one_hot
from torch.distributions import Normal
from mne.filter import notch_filter
from mne.channels.interpolation import _make_interpolation_matrix
from mne.channels import make_standard_montage
import matplotlib.pyplot as plt
from ecgdetectors import Detectors
from scipy.interpolate import CubicSpline
from scipy.ndimage.interpolation import shift
from scipy.signal import butter, lfilter
'''
ECG transfrom from "Effective Data Augmentation, Filters, and Automation Techniques 
for Automatic 12-Lead ECG Classification Using Deep Residual Neural Networks"
Junmo An, Member, IEEE, Richard E. Gregg, and Soheil Borhani 
2022 44th Annual International Conference of
the IEEE Engineering in Medicine & Biology Society (EMBC)
Scottish Event Campus, Glasgow, UK, July 11-15, 2022
Reimplement
'''


def identity(x, *args, **kwargs):
    return x

def Amplifying(x,magnitude, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    factor = rng.normal(loc=1., scale=magnitude, size=(x.shape[0],1,1)) #diff batch
    x = x.detach().cpu().numpy()
    new_x = np.multiply(x, factor)
    new_x = torch.from_numpy(new_x).float()
    return new_x

def Baseline_wander(x,magnitude,seq_len=None,sfreq=100, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    batch_size, n_channels, max_seq_len = x.shape
    if seq_len==None: #!!!bug when max_seq!=seq
        seq_len = max_seq_len
    rd_start = rng.uniform(0, 2*np.pi, size=(batch_size, 1))
    rd_hz = rng.uniform(0, 0.5, size=(batch_size, 1))
    tot_s = seq_len / sfreq
    rd_T = tot_s / rd_hz
    factor = np.linspace(rd_start,rd_start + (2*np.pi / rd_T),seq_len,axis=-1) #(bs,len) ?
    sin_wave = magnitude * np.sin(factor)[:,np.newaxis,:]
    x = x.detach().cpu().numpy()
    new_x = x + sin_wave
    new_x = torch.from_numpy(new_x).float()
    return new_x

def chest_leads_shuffle():
    pass

def dropout(x,magnitude, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    batch_size, n_channels, max_seq_len = x.shape
    mask = rng.binomial(n=1,p=magnitude,size=(batch_size,1,max_seq_len))
    x = x.detach().cpu().numpy()
    new_x = np.multiply(x, mask)
    new_x = torch.from_numpy(new_x).float()
    return new_x

#cutout=time_mask
#Gaussian noise addition already have
#Horizontal flip already have
#Lead removal=channel drop

def Lead_reversal(x, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    batch_size, n_channels, max_seq_len = x.shape
    order = np.arange(n_channels)[::-1]
    x = x.detach().cpu().numpy()
    new_x = x[:,order,:]
    new_x = torch.from_numpy(new_x).float()
    return new_x

#Leads order shuffling=channel shuffle

def Line_noise(x,magnitude,seq_len=None,sfreq=100, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    batch_size, n_channels, max_seq_len = x.shape
    if seq_len==None: #!!!bug when max_seq!=seq
        seq_len = max_seq_len
    rd_start = rng.uniform(0, 2*np.pi, size=(batch_size, 1))
    rd_hz = np.ones((batch_size, 1)) * 60.0
    tot_s = seq_len / sfreq
    rd_T = tot_s / rd_hz
    factor = np.linspace(rd_start,rd_start + (2*np.pi / rd_T),seq_len,axis=-1) #(bs,len) ?
    sin_wave = magnitude * np.sin(factor)[:,np.newaxis,:]
    x = x.detach().cpu().numpy()
    new_x = x + sin_wave
    new_x = torch.from_numpy(new_x).float()
    return new_x

#Scaling already have

#Time-window shifting
def Time_shift(x,magnitude,seq_len=None,sfreq=100, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    batch_size, n_channels, max_seq_len = x.shape
    if seq_len==None: #!!!bug when max_seq!=seq
        seq_len = max_seq_len
    shift_val = rng.uniform(-magnitude*sfreq, magnitude*sfreq, size=(batch_size, 1))[:,np.newaxis,:].int()
    x = x.detach().cpu().numpy()
    new_x = shift(x,shift_val,cval=0.0)
    new_x = torch.from_numpy(new_x).float()
    return new_x

#Vertical flip already have

#filters
def butter_bandpass(lowcut, highcut, fs, order=3):
    return butter(order, [lowcut, highcut], fs=fs, btype='bandpass')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(lowcut, fs, order=3):
    return butter(order, lowcut, fs=fs, btype='lowpass')
def butter_highpass(lowcut, fs, order=3):
    return butter(order, lowcut, fs=fs, btype='highpass')