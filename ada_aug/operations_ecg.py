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
from scipy.signal import butter, lfilter, iirnotch
from scipy.ndimage import gaussian_filter1d
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
    factor = np.clip(rng.normal(loc=1., scale=magnitude, size=(x.shape[0],1,1)),0,10) #diff batch
    x = x.detach().cpu().numpy()
    new_x = np.multiply(x, factor)
    new_x = torch.from_numpy(new_x).float()
    return new_x

def Baseline_wander(x,magnitude,seq_len=None,sfreq=100, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    batch_size, n_channels, max_seq_len = x.shape
    if seq_len==None: #!!!bug when max_seq!=seq
        seq_len = max_seq_len
    tot_waves = np.zeros((batch_size, 1, max_seq_len))
    hz_list = [0.05, 0.1, 0.15, 0.2, 0.5]
    for i in range(5):
        rd_start = rng.uniform(0, 2*np.pi, size=(batch_size, 1))
        rd_hz = np.ones((batch_size, 1)) * hz_list[i]
        tot_s = seq_len / sfreq
        rd_T = tot_s * rd_hz
        factor = np.linspace(rd_start,np.add(rd_start, (2*np.pi * rd_T)),seq_len,axis=-1) #(bs,len) ?
        sin_wave = magnitude * np.sin(factor)
        tot_waves[:,:,:seq_len] += sin_wave
    
    x = x.detach().cpu().numpy()
    new_x = x + sin_wave
    new_x = torch.from_numpy(new_x).float()
    return new_x

def chest_leads_shuffle(x,magnitude, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    batch_size, n_channels, max_seq_len = x.shape
    order = np.arange(n_channels)[::-1]
    rng.shuffle(order[6:])
    x = x.detach().cpu().numpy()
    new_x = x[:,order,:]
    new_x = torch.from_numpy(new_x).float()
    return new_x
    
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

def Lead_reversal(x,magnitude, random_state=None, *args, **kwargs):
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
    rd_T = tot_s * rd_hz
    factor = np.linspace(rd_start,np.add(rd_start, (2*np.pi * rd_T)),seq_len,axis=-1) #(bs,1,len) ?
    sin_wave = magnitude * np.sin(factor)
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
    shift_val = rng.uniform(-magnitude*seq_len, magnitude*seq_len, size=(batch_size)).astype(int)
    x = x.detach().cpu().numpy()
    new_x = []
    for (e_x,e_shift) in zip(x,shift_val):
        o_x = shift(e_x,[0,shift_val],cval=0.0)
        new_x.append(o_x)
    new_x = torch.from_numpy(np.array(new_x)).float()
    return new_x

#Vertical flip already have

#large-amplitude waveforms
def _sample_mask_start(X, mask_len_samples, random_state, seq_len=None):
    rng = check_random_state(random_state)
    #seq_length = torch.as_tensor(X.shape[-1], device=X.device)
    mask_start = torch.as_tensor(rng.uniform(
        low=0, high=1, size=X.shape[0],
    ), device=X.device) * (seq_len - mask_len_samples)
    return mask_start
def _saturation_time(X, mask_start_per_sample, mask_len_samples, val=0):
    #mask = torch.ones_like(X)
    for i, start in enumerate(mask_start_per_sample):
        X[i, :, int(start):int(start + mask_len_samples)] = val #every channel
    return X
def random_time_saturation(x, mask_len_samples, random_state=None,seq_len=None,sfreq=100, *args, **kwargs):
    mask_len_samples = mask_len_samples * sfreq
    mask_start = _sample_mask_start(x, mask_len_samples, random_state,seq_len=seq_len)
    max_val = x.max()
    x = x.detach().cpu().clone()
    return _saturation_time(x, mask_start, mask_len_samples,max_val)

#filters
def butter_bandpass(lowcut, highcut, fs, order=3):
    return butter(order, [lowcut, highcut], fs=fs, btype='bandpass')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter(data,lowcut, fs, order=3):
    b, a = butter(order, lowcut, fs=fs, btype='lowpass')
    y = lfilter(b, a, data)
    return y
def butter_highpass_filter(data,lowcut, fs, order=3):
    b, a = butter(order, lowcut, fs=fs, btype='highpass')
    y = lfilter(b, a, data)
    return y
#0.5~47
def Band_pass(x,magnitude,seq_len=None,sfreq=100, random_state=None, *args, **kwargs):
    low_freq = 0.5
    high_freq = 47
    batch_size, n_channels, max_seq_len = x.shape
    x = x.detach().cpu().numpy().reshape((batch_size*n_channels,max_seq_len))
    new_x = butter_bandpass_filter(x,low_freq,high_freq,fs=sfreq)
    new_x = new_x.reshape((batch_size, n_channels, max_seq_len))
    new_x = torch.from_numpy(new_x).float()
    return new_x
#k=3
def Gaussian_blur(x,magnitude,seq_len=None,sfreq=100, random_state=None, *args, **kwargs):
    truncate = 3.0
    sigma = 1 / 6.0
    batch_size, n_channels, max_seq_len = x.shape
    x = x.detach().cpu().numpy().reshape((batch_size*n_channels,max_seq_len))
    new_x = gaussian_filter1d(x,sigma=sigma,truncate=truncate)
    new_x = new_x.reshape((batch_size, n_channels, max_seq_len))
    new_x = torch.from_numpy(new_x).float()
    return new_x
#high pass
def High_pass(x,magnitude,seq_len=None,sfreq=100, random_state=None, *args, **kwargs):
    low_freq = 0.5
    batch_size, n_channels, max_seq_len = x.shape
    x = x.detach().cpu().numpy().reshape((batch_size*n_channels,max_seq_len))
    new_x = butter_highpass_filter(x,low_freq,fs=sfreq)
    new_x = new_x.reshape((batch_size, n_channels, max_seq_len))
    new_x = torch.from_numpy(new_x).float()
    return new_x
#low pass
def Low_pass(x,magnitude,seq_len=None,sfreq=100, random_state=None, *args, **kwargs):
    low_freq = 47.0
    batch_size, n_channels, max_seq_len = x.shape
    x = x.detach().cpu().numpy().reshape((batch_size*n_channels,max_seq_len))
    new_x = butter_lowpass_filter(x,low_freq,fs=sfreq)
    new_x = new_x.reshape((batch_size, n_channels, max_seq_len))
    new_x = torch.from_numpy(new_x).float()
    return new_x
#IIR notch Q=30, 60Hz
def IIR_notch(x,magnitude,seq_len=None,sfreq=100, random_state=None, *args, **kwargs):
    low_freq = 60.0
    w0 = min(low_freq, sfreq/2)
    batch_size, n_channels, max_seq_len = x.shape
    x = x.detach().cpu().numpy().reshape((batch_size*n_channels,max_seq_len))
    b, a = iirnotch(w0, Q=30, fs=sfreq)
    new_x = lfilter(b, a, x)
    new_x = new_x.reshape((batch_size, n_channels, max_seq_len))
    new_x = torch.from_numpy(new_x).float()
    return new_x
#Sigmoid compress x
def Sigmoid_compress(x, *args, **kwargs):
    new_x = x.detach().clone().cpu()
    return torch.sigmoid(new_x)

