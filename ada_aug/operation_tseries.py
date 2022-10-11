# code is adapted from CADDA and braincode
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

#for model: (len, channel)
#for this file (channel, len)!
def identity(x, *args, **kwargs):
    return x
def time_reverse(X, *args, **kwargs):
    return torch.flip(X, [-1])
def sign_flip(X, *args, **kwargs):
    return -X
def downsample_shift_from_arrays(X, factor, offset, *args, **kwargs):
    return X[..., offset::factor]

def _new_random_fft_phase_odd(n, device, random_state=None):
    rng = check_random_state(random_state)
    random_phase = torch.from_numpy(
        2j * np.pi * rng.random((n - 1) // 2)
    ).to(device)
    return torch.cat([
        torch.as_tensor([0.0], device=device),
        random_phase,
        -torch.flip(random_phase, [-1])
    ])
def _new_random_fft_phase_even(n, device, random_state=None):
    rng = check_random_state(random_state)
    random_phase = torch.from_numpy(
        2j * np.pi * rng.random(n // 2 - 1)
    ).to(device)
    return torch.cat([
        torch.as_tensor([0.0], device=device),
        random_phase,
        torch.as_tensor([0.0], device=device),
        -torch.flip(random_phase, [-1])
    ])
_new_random_fft_phase = {
    0: _new_random_fft_phase_even,
    1: _new_random_fft_phase_odd
}
def _fft_surrogate(x=None, f=None, eps=1, random_state=None):
    """ FT surrogate augmentation of a single EEG channel, as proposed in [1]_

    Function copied from https://github.com/cliffordlab/sleep-convolutions-tf
    and modified.

    MIT License

    Copyright (c) 2018 Clifford Lab

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

    Parameters
    ----------
    x: torch.tensor, optional
        Single EEG channel signal in time space. Should not be passed if f is
        given. Defaults to None.
    f: torch.tensor, optional
        Fourier spectrum of a single EEG channel signal. Should not be passed
        if x is given. Defaults to None.
    eps: float, optional
        Float between 0 and 1 setting the range over which the phase
        pertubation is uniformly sampled: [0, `eps` * 2 * `pi`]. Defaults to 1.
    random_state: int | numpy.random.Generator, optional
        By default None.

    References
    ----------
    .. [1] Schwabedal, J. T., Snyder, J. C., Cakmak, A., Nemati, S., &
       Clifford, G. D. (2018). Addressing Class Imbalance in Classification
       Problems of Noisy Signals by using Fourier Transform Surrogates. arXiv
       preprint arXiv:1806.08675.
    """
    assert isinstance(
        eps,
        (Real, torch.FloatTensor, torch.cuda.FloatTensor)
    ) and 0 <= eps <= 1, f"eps must be a float beween 0 and 1. Got {eps}."
    if f is None:
        assert x is not None, 'Neither x nor f provided.'
        f = fft(x.double(), dim=-1)
        device = x.device
    else:
        device = f.device
    n = f.shape[-1]
    random_phase = _new_random_fft_phase[n % 2](
        n,
        device=device,
        random_state=random_state
    )
    f_shifted = f * torch.exp(eps * random_phase)
    shifted = ifft(f_shifted, dim=-1)
    return shifted.real.float()
def fft_surrogate(X, magnitude, random_state, *args, **kwargs):
    transformed_X = _fft_surrogate( #single channel???
        x=X,
        eps=magnitude,
        random_state=random_state
    )
    return transformed_X


def _pick_channels_randomly(X, magnitude, random_state):
    rng = check_random_state(random_state)
    batch_size, n_channels, _ = X.shape
    # allows to use the same RNG
    unif_samples = torch.as_tensor(
        rng.uniform(0, 1, size=(batch_size, n_channels)),
        dtype=torch.float,
        device=X.device,
    )
    # equivalent to a 0s and 1s mask, but allows to backprop through
    # could also be done using torch.distributions.RelaxedBernoulli
    return torch.sigmoid(1000 * (unif_samples - magnitude)).to(X.device)
def channel_dropout(X, magnitude, random_state=None, *args, **kwargs):
    mask = _pick_channels_randomly(X, magnitude, random_state)
    return X * mask.unsqueeze(-1)

def _make_permutation_matrix(X, mask, random_state):
    rng = check_random_state(random_state)
    batch_size, n_channels, _ = X.shape
    hard_mask = mask.round()
    batch_permutations = torch.empty(
        batch_size, n_channels, n_channels, device=X.device
    )
    for b, mask in enumerate(hard_mask):
        channels_to_shuffle = torch.arange(n_channels)
        channels_to_shuffle = channels_to_shuffle[mask.bool()]
        channels_permutation = np.arange(n_channels)
        channels_permutation[channels_to_shuffle] = rng.permutation(
            channels_to_shuffle
        )
        channels_permutation = torch.as_tensor(
            channels_permutation, dtype=torch.int64, device=X.device
        )
        batch_permutations[b, ...] = one_hot(channels_permutation)
    return batch_permutations
def channel_shuffle(X, magnitude, random_state=None, *args, **kwargs):
    mask = _pick_channels_randomly(X, 1 - magnitude, random_state)
    batch_permutations = _make_permutation_matrix(X, mask, random_state)
    return torch.matmul(batch_permutations, X)

def add_gaussian_noise(X, std, random_state=None, *args, **kwargs):
    # XXX: Maybe have rng passed as argument here
    rng = check_random_state(random_state)
    noise = torch.from_numpy(
        rng.normal(
            loc=np.zeros(X.shape),
            scale=1
        )
    ).float().to(X.device) * std
    transformed_X = X + noise
    return transformed_X

def exp_add_gaussian_noise(X, std, random_state=None, *args, **kwargs):
    # XXX: Maybe have rng passed as argument here
    rng = check_random_state(random_state)
    noise = torch.from_numpy(
        rng.normal(
            loc=np.zeros(X.shape),
            scale=1
        )
    ).float().to(X.device) * std
    transformed_X = X + noise
    return transformed_X

def permute_channels(X, permutation, *args, **kwargs):
    return X[..., permutation, :]


def _sample_mask_start(X, mask_len_samples, random_state, seq_len=None):
    rng = check_random_state(random_state)
    #seq_length = torch.as_tensor(X.shape[-1], device=X.device)
    mask_start = torch.as_tensor(rng.uniform(
        low=0, high=1, size=X.shape[0],
    ), device=X.device) * (seq_len - mask_len_samples)
    return mask_start
def _mask_time(X, mask_start_per_sample, mask_len_samples):
    mask = torch.ones_like(X)
    for i, start in enumerate(mask_start_per_sample):
        mask[i, :, int(start):int(start) + mask_len_samples] = 0 #every channel
    return X * mask
def _relaxed_mask_time(X, mask_start_per_sample, mask_len_samples):
    batch_size, n_channels, seq_len = X.shape
    t = torch.arange(seq_len, device=X.device).float()
    t = t.repeat(batch_size, n_channels, 1)
    mask_start_per_sample = mask_start_per_sample.view(-1, 1, 1)
    s = 1000 * 10**-np.log10(seq_len)
    mask = 2 - (
        torch.sigmoid(s * (t - mask_start_per_sample)) +
        torch.sigmoid(s * -(t - mask_start_per_sample - mask_len_samples))
    ).float().to(X.device)
    return X * mask
def random_time_mask(X, mask_len_samples, random_state=None,seq_len=None,sfreq=100, *args, **kwargs):
    mask_len_samples = mask_len_samples * sfreq
    mask_start = _sample_mask_start(X, mask_len_samples, random_state,seq_len=seq_len)
    return _relaxed_mask_time(X, mask_start, mask_len_samples)
def exp_time_mask(X, mask_len_samples, random_state=None,seq_len=None, *args, **kwargs):
    #seq_len = X.shape[2]
    all_mask_len_samples = int(seq_len * mask_len_samples / 100.0)
    mask_start = _sample_mask_start(X, all_mask_len_samples, random_state,seq_len=seq_len)
    return _mask_time(X, mask_start, all_mask_len_samples)
def _sample_mask_start_info(X, mask_len_samples,start,end, random_state):
    rng = check_random_state(random_state)
    seq_length = X.shape[-1]
    #print(f'start:{start} ,end:{end} ,masklen:{mask_len_samples}')
    c_start = max(min(start,end-mask_len_samples),0)
    c_end = min(end,seq_length-mask_len_samples+1)
    mask_start = torch.as_tensor(rng.randint(
        low=c_start, high=c_end, size=X.shape[0],
    ), device=X.device)
    return mask_start
def info_time_mask(X, mask_len_samples,start,end,
        random_state=None, *args, **kwargs):
    seq_len = X.shape[2]
    all_mask_len_samples = int(seq_len * mask_len_samples / 100.0)
    #calculate start/end
    mask_start = _sample_mask_start_info(X, all_mask_len_samples,start,end, random_state)
    return _mask_time(X, mask_start, all_mask_len_samples)

def random_bandstop(X, bandwidth, max_freq=50, sfreq=100, random_state=None, *args,
                    **kwargs):
    rng = check_random_state(random_state)
    transformed_X = X.clone()
    # Prevents transitions from going below 0 and above max_freq
    notched_freqs = rng.uniform(
        low=1 + 2 * bandwidth,
        high=max_freq - 1 - 2 * bandwidth,
        size=X.shape[0]
    )
    # I just worry that this might be to complex for gradient descent and
    # it would be a shame to make a straight-through here... A new version
    # using torch convolution might be necessary...
    for c, (sample, notched_freq) in enumerate(
            zip(transformed_X, notched_freqs)):
        sample = sample.detach().cpu().numpy().astype(np.float64)
        transformed_X[c] = torch.as_tensor(notch_filter(
            sample,
            Fs=sfreq,
            freqs=notched_freq,
            method='fir',
            notch_widths=bandwidth,
            verbose=False
        ))
    return transformed_X

def exp_bandstop(X, bandwidth, max_freq=50, sfreq=100, random_state=None, *args, #300 is 2 * max ecg
                    **kwargs):
    rng = check_random_state(random_state)
    transformed_X = X.clone()
    # Prevents transitions from going below 0 and above max_freq
    notched_freqs = rng.uniform(
        low=1 + bandwidth/2,
        high=max_freq - 1 - bandwidth/2,
        size=X.shape[0]
    )
    # I just worry that this might be to complex for gradient descent and
    # it would be a shame to make a straight-through here... A new version
    # using torch convolution might be necessary...
    for c, (sample, notched_freq) in enumerate(zip(transformed_X, notched_freqs)):
        sample = sample.cpu().numpy().astype(np.float64)
        transformed_X[c] = torch.as_tensor(notch_filter(
            sample,
            Fs=sfreq,
            freqs=notched_freq,
            method='fir',
            notch_widths=bandwidth,
            verbose=False
        ))
    return transformed_X

def hilbert_transform(x):
    if torch.is_complex(x):
        raise ValueError("x must be real.")

    N = x.shape[-1]
    f = fft(x, N, dim=-1)
    h = torch.zeros_like(f)
    if N % 2 == 0:
        h[..., 0] = h[..., N // 2] = 1
        h[..., 1:N // 2] = 2
    else:
        h[..., 0] = 1
        h[..., 1:(N + 1) // 2] = 2

    return ifft(f * h, dim=-1)
def nextpow2(n):
    """Return the first integer N such that 2**N >= abs(n)"""
    return int(np.ceil(np.log2(np.abs(n))))
def _freq_shift(x, fs, f_shift):
    """
    Shift the specified signal by the specified frequency.

    See https://gist.github.com/lebedov/4428122
    """
    # Pad the signal with zeros to prevent the FFT invoked by the transform
    # from slowing down the computation:
    n_channels, N_orig = x.shape[-2:]
    N_padded = 2 ** nextpow2(N_orig)
    t = torch.arange(N_padded, device=x.device) / fs
    padded = pad(x, (0, N_padded - N_orig))
    analytical = hilbert_transform(padded)
    if isinstance(f_shift, (float, int, np.ndarray, list)):
        f_shift = torch.as_tensor(f_shift).float()
    reshaped_f_shift = f_shift.repeat(
        N_padded, n_channels, 1).T
    shifted = analytical * torch.exp(2j * np.pi * reshaped_f_shift * t)
    return shifted[..., :N_orig].real.float()
def freq_shift(X, max_shift, sfreq=100, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    delta_freq = torch.as_tensor(
        rng.uniform(size=X.shape[0]), device=X.device) * max_shift
    transformed_X = _freq_shift(
        x=X,
        fs=sfreq,
        f_shift=delta_freq,
    )
    return transformed_X
def exp_freq_shift(X, max_shift, sfreq=100, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    delta_freq = torch.as_tensor(
        rng.uniform(size=X.shape[0]), device=X.device) * max_shift
    transformed_X = _freq_shift(
        x=X,
        fs=sfreq,
        f_shift=delta_freq,
    )
    return transformed_X
'''
Delete EEG special transfroms
'''

### for ECG transfrom
#tseries signal (batch,channel,len)
def RR_permutation(X, magnitude,sfreq=100, random_state=None, *args, **kwargs): #x:(batch,channel,len)
    """
    From 'Nita, Sihem, et al. 
    "A new data augmentation convolutional neural network for human emotion recognition based on ECG signals."
    Biomedical Signal Processing and Control 75 (2022): 103580.'
    Using https://github.com/berndporr/py-ecg-detectors module for detection
    """
    x = X.detach().cpu().numpy()
    num_sample, num_leads, num_len = x.shape
    detectors = Detectors(sfreq) #need input ecg: (seq_len)
    rng = check_random_state(random_state)
    #select_lead = rng.randint(0, num_leads-1)
    select_lead = 0 #!!!tmp
    rpeaks_array = detectors.two_average_detector(x[0,select_lead,:])
    seg_ids = [i for i in range(len(rpeaks_array)-1)]
    permut_seg_ids = rng.permutation(seg_ids)
    seg_list = []
    start_point = 0
    for end_point in rpeaks_array + [num_len]:
        seg_list.append(x[:,:,start_point:end_point])
        start_point = end_point
    #permutation
    perm_seg_list = []
    for i in range(len(seg_list)):
        if i==0 or i==len(seg_list)-1:
            perm_seg_list.append(seg_list[i])
        else:
            perm_seg_list.append(seg_list[permut_seg_ids[i-1]+1])
    #concat & back tensor
    new_x = np.concatenate(perm_seg_list,axis=2)
    new_x = torch.from_numpy(new_x).float()
    return new_x
#tseries signal (batch,channel,len)
def QRS_resample(X, magnitude,sfreq=100, random_state=None, *args, **kwargs):
    """
    From 'A novel data augmentation method to enhance deep neural networks for detection of atrial fibrillation.
    PingCao et.al. Biomedical Signal Processing and Control Volume 56, February 2020, 101675'
    QRS detection using 'Pan, J., Tompkins, W.J.: A real-time QRS detection algorithm. IEEE Trans. Biomed. Eng. 32, 230-236 (1985)'
    Using https://github.com/berndporr/py-ecg-detectors module for detection
    """
    detectors = Detectors(sfreq) #need input ecg: (seq_len)
    qrs_interval = 0 #int(0.1 * sfreq)
    x = X.detach().cpu().numpy()
    num_sample, num_leads, num_len = x.shape
    window_size = num_len
    rng = check_random_state(random_state)
    #select_lead = rng.randint(0, num_leads-1)
    select_lead = 0 #!!!tmp
    rpeaks_array = detectors.pan_tompkins_detector(x[0,select_lead,:])
    first_p,last_p = int(max(rpeaks_array[0] - qrs_interval/2,0)), int(rpeaks_array[-1] - qrs_interval/2)
    dup_x = np.concatenate([x[:,:,first_p:last_p],x[:,:,first_p:last_p]],axis=2)
    if dup_x.shape[2] >= window_size: #if long enough
        window_start = rng.randint(0, dup_x.shape[2] - window_size)
        new_x = dup_x[:,:,window_start:window_start+window_size]
        new_x = torch.from_numpy(new_x).float()
    else:
        new_x = X.detach().cpu()
    return new_x
'''
4 METHODS FROM "Data Augmentation for Deep Learning-Based ECG Analysis"
'''
#Window Slicing: tseries signal (batch,channel,len)
#WW, WS
def window_slice(x,rng, reduce_ratio=0.9,start=0,end=None): #ref (batch, time_steps, channel)
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = max(np.ceil(reduce_ratio*x.shape[1]).astype(int),1)
    if target_len >= x.shape[1]:
        return x
    if end==None:
        end = x.shape[1]-target_len
    else:
        end = end - target_len
    if start>0:
        start = min(start,end-1)
    #print(f'start:{start} ,end:{end} ,masklen:{target_len}')
    starts = rng.randint(low=max(start,0), high=max(end,1), size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret
def Window_Slicing(X, magnitude, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    x = X.permute(0,2,1).detach().cpu().numpy()
    new_x = window_slice(x,rng,1. - magnitude)
    new_x = torch.from_numpy(new_x).float().permute(0,2,1) #back
    return new_x
def info_Window_Slicing(X, magnitude,start,end, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    x = X.permute(0,2,1).detach().cpu().numpy()
    new_x = window_slice(x,rng,1. - magnitude,start=start,end=end)
    new_x = torch.from_numpy(new_x).float().permute(0,2,1) #back
    return new_x

def Window_Slicing_Circle(X, magnitude, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    window_size = X.shape[2]
    x = X.detach().cpu().numpy()
    dup_x = np.concatenate([x,x],axis=2)
    window_start = rng.randint(0, dup_x.shape[2] - window_size)
    new_x = dup_x[:,:,window_start:window_start+window_size]
    new_x = torch.from_numpy(new_x).float()
    return new_x
#TS_Permutation: tseries signal (batch,channel,len)
def TS_Permutation(X, magnitude, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    N_clip = rng.randint(1, int(magnitude), size=None)
    seg_ids = [i for i in range(N_clip)]
    x = X.detach().cpu().numpy()
    num_sample, num_leads, num_len = x.shape
    seg_len = int(num_len/N_clip)
    permut_seg_ids = rng.permutation(seg_ids)
    seg_list = []
    start_point = 0
    for i in range(N_clip-1):
        seg_list.append(x[:,:,start_point:start_point+seg_len])
        start_point = start_point+seg_len
    seg_list.append(x[:,:,start_point:]) #add last seg
    #permutation
    perm_seg_list = []
    for i in range(len(seg_list)):
        perm_seg_list.append(seg_list[permut_seg_ids[i]])
    #concat & back tensor
    new_x = np.concatenate(perm_seg_list,axis=2)
    new_x = torch.from_numpy(new_x).float()
    return new_x
#Concat_Resample same as Window Slicing
#Time Warp using generalize time-series module
def time_warp(x,rng, sigma=0.2, knot=4): #ref (batch, time_steps, channel)
    orig_steps = np.arange(x.shape[1])
    random_warps = rng.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, 1)) #modify for same warp between channels
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,0])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return ret
def Time_Warp(X, magnitude, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    x = X.permute(0,2,1).detach().cpu().numpy()
    new_x = time_warp(x,rng,magnitude)
    new_x = torch.from_numpy(new_x).float().permute(0,2,1) #back
    return new_x

'''
METHODS FROM Time Series Data Augmentation github
"https://github.com/uchidalab/time_series_augmentation" and
"https://github.com/timeseriesAI/tsai"
Choose method appear in ECGAug: Translation, Scaling Voltage, Time Warping, 
Reflection(reflect by q end)=>no for all ts, Adding Noise: high=>random noise, low=>baseline wander
'''
#add a scalar
def Translation(X, magnitude, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    amp = rng.uniform(low=-magnitude, high=magnitude, size=(x.shape[0],x.shape[1]))
    x = X.detach().cpu()
    return x + amp[:,:,np.newaxis]
#scaling voltage
def scaling(x,rng, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = rng.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[1])) #diff batch&channel
    return np.multiply(x, factor[:,:,np.newaxis])
def Scaling(X, magnitude, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    x = X.detach().cpu().numpy()
    new_x = scaling(x,rng,magnitude)
    new_x = torch.from_numpy(new_x).float()
    return new_x
#Reflection already have sign_flip
#Adding Noise: low=>baseline wander
def magnitude_warp(x, rng, sigma=0.2, knot=4): #ref (batch, time_steps, channel)
    orig_steps = np.arange(x.shape[1])
    random_warps = rng.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper
    return ret
def Magnitude_Warp(X, magnitude, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    x = X.permute(0,2,1).detach().cpu().numpy()
    new_x = magnitude_warp(x,rng,magnitude)
    new_x = torch.from_numpy(new_x).float().permute(0,2,1) #back
    return new_x
'''
Common Time Series Augmentation from
"https://github.com/uchidalab/time_series_augmentation" and
T. T. Um et al, "Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks," in ACM ICMI, pp. 216-220, 2017.
'''

def window_warp(x,rng, window_ratio=0.1, scales=[0.5, 2.],start=1,end=None): #ref (batch, time_steps, channel)
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = rng.choice(scales, x.shape[0])
    warp_size = min(np.ceil(window_ratio*x.shape[1]).astype(int),x.shape[1]-1)
    if warp_size<=1:
        return x
    window_steps = np.arange(warp_size)
    if end==None:
        end = x.shape[1]-warp_size-1
    else:
        end = end - warp_size - 1
    if start>1:
        start = min(start,end-1)
    #print(f'start:{start} ,end:{end} ,masklen:{warp_size}')
    window_starts = rng.randint(low=max(start,1), high=max(end,2), size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret
def Window_Warp(X, magnitude, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    x = X.permute(0,2,1).detach().cpu().numpy()
    new_x = window_warp(x,rng,magnitude)
    new_x = torch.from_numpy(new_x).float().permute(0,2,1) #back
    return new_x
def info_Window_Warp(X, magnitude,start,end, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    x = X.permute(0,2,1).detach().cpu().numpy()
    new_x = window_warp(x,rng,magnitude,start=start,end=end)
    new_x = torch.from_numpy(new_x).float().permute(0,2,1) #back
    return new_x
#Advance Method not include here

TS_OPS_NAMES = [
    'identity', #identity
    'time_reverse', #time reverse
    'fft_surrogate',
    'channel_dropout',
    'channel_shuffle',
    # 'channel-sym', this is only for eeg
    'random_time_mask',
    'add_gaussian_noise',
    'random_bandstop',
    'sign_flip',
    'freq_shift',
    # 'rotz', this is only for eeg
    # 'roty', this is only for eeg
    # 'rotx', this is only for eeg
]
TS_AUGMENT_LIST = [
        (identity, 0, 1),  # 0
        (time_reverse, 0, 1),  # 1
        (fft_surrogate, 0, 1),  # 2
        (channel_dropout, 0, 1),  # 3
        (channel_shuffle, 0, 1),  # 4
        (random_time_mask, 0, 1),  # 5 impl
        (add_gaussian_noise, 0, 0.2),  # 6
        (random_bandstop, 0, 2),  # 7
        (sign_flip, 0, 1),  # 8
        (freq_shift, 0, 5),  # 9
        ]
ECG_OPS_NAMES = [
    'RR_permutation',
    'QRS_resample',
]
ECG_AUGMENT_LIST = [
    (RR_permutation, 0, 1),
    (QRS_resample, 0, 1),
]
TS_ADD_NAMES = [
    'Window_Slicing',
    'Window_Slicing_Circle',
    'TS_Permutation',
    'Time_Warp',
    'Scaling',
    'Magnitude_Warp',
    'Window_Warp',
]
TS_ADD_LIST = [
    (Window_Slicing, 0, 1),  # 0
    (Window_Slicing_Circle, 0, 1),  # 1
    (TS_Permutation, 2, 20),  # 2
    (Time_Warp, 0, 0.2),  # 3
    (Scaling, 0, 1),  # 4
    (Magnitude_Warp, 0, 0.4),  # 5
    (Window_Warp, 0, 1),  # 6
]
MAG_TEST_NAMES = [
    'fft_surrogate',
    'channel_dropout',
    'channel_shuffle',
    'random_time_mask',
    'add_gaussian_noise',
    'random_bandstop',
    'freq_shift',
    'Window_Slicing',
    'TS_Permutation',
    'Time_Warp',
    'Scaling',
    'Magnitude_Warp',
    'Window_Warp',
]
NOMAG_TEST_NAMES = [
    'time_reverse', #time reverse
    'sign_flip',
    'RR_permutation',
    'QRS_resample',
    'Window_Slicing_Circle',
]
TS_EXP_LIST = [
    (exp_time_mask, 0, 100),
    (exp_bandstop, 0, 48), #sample freq=100, bandstop=48 because of notch
    (exp_freq_shift, 0, 10), #sample freq=100
    (exp_add_gaussian_noise, 0, 1),  # noise up to std
]
EXP_TEST_NAMES =[
    'exp_time_mask',
    'exp_bandstop',
    'exp_freq_shift',
    'exp_add_gaussian_noise',
]
INFO_EXP_LIST = [
    (info_time_mask, 0, 100),
    (info_Window_Warp, 0, 1),
    (info_Window_Slicing, 0, 1),
]
INFO_TEST_NAMES =[
    'info_time_mask',
    'info_Window_Warp',
    'info_Window_Slicing',
]

AUGMENT_DICT = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in TS_AUGMENT_LIST+ECG_AUGMENT_LIST+TS_ADD_LIST+TS_EXP_LIST+INFO_EXP_LIST}
selopt = ['cut','paste']
SELECTIVE_DICT = {
    'identity':selopt[1], #identity
    'time_reverse':selopt[1], #!undefine
    'fft_surrogate':selopt[1],
    'channel_dropout':selopt[1],
    'channel_shuffle':selopt[1],
    'random_time_mask':selopt[0],
    'add_gaussian_noise':selopt[1],
    'random_bandstop':selopt[1],
    'sign_flip':selopt[1],
    'freq_shift':selopt[1],
    'Window_Slicing':selopt[0],
    'Window_Slicing_Circle':selopt[1],
    'TS_Permutation':selopt[1], #!undefine
    'Time_Warp':selopt[1],
    'Scaling':selopt[1],
    'Magnitude_Warp':selopt[1],
    'Window_Warp':selopt[0], #!problems
    'RR_permutation':selopt[1], #!undefine
    'QRS_resample':selopt[1], #!undefine
}
def get_augment(name):
    return AUGMENT_DICT[name]

def apply_augment(img, name, level, rd_seed=None,sfreq=100,seq_len=None):
    augment_fn, low, high = get_augment(name)
    assert 0 <= level
    assert level <= 1
    #change tseries signal from (len,channel) to (batch,channel,len)
    max_seq_len , channel = img.shape
    if seq_len==None: #assume aug_img != img
        seq_len = max_seq_len
    img = img.permute(1,0).view(1,channel,max_seq_len)
    tmp_img = img[:,:,:seq_len]
    aug_value = level * (high - low) + low
    #print('Device: ',aug_value.device)
    aug_img = augment_fn(tmp_img, aug_value,random_state=rd_seed,sfreq=sfreq,seq_len=seq_len)
    img[:,:,:seq_len] = aug_img #tmp fix, may become slower
    return img.permute(0,2,1).detach().view(max_seq_len,channel) #back to (len,channel)

def plot_line(t,x,title=None):
    plt.clf()
    channel_num = x.shape[-1]
    for i in  range(channel_num):
        plt.plot(t, x[:,i])
    if title:
        plt.title(title)
    plt.show()

def lt(a,b):
    return a < b
def ge(a,b):
    return a >= b
def le(a,b):
    return a <= b

class ToTensor:
    def __init__(self) -> None:
        pass
    def __call__(self, img):
        return torch.tensor(img).float()

class RandAugment:
    def __init__(self, n, m, rd_seed=None,augselect='',sfreq=100):
        self.n = n
        self.m = m      # [0, 1]
        self.augment_list = TS_AUGMENT_LIST
        if 'tsadd' in augselect:
            print('Augmentation add TS_ADD_LIST')
            self.augment_list += TS_ADD_LIST
        if 'ecg' in augselect:
            print('Augmentation add ECG_AUGMENT_LIST')
            self.augment_list += ECG_AUGMENT_LIST
        self.augment_ids = [i for i in range(len(self.augment_list))]
        self.rng = check_random_state(rd_seed)
        self.sfreq = sfreq
    def __call__(self, img, seq_len=None):
        #print(img.shape)
        max_seq_len , channel = img.shape
        if seq_len==None:
            seq_len = max_seq_len
        img = img.permute(1,0).view(1,channel,max_seq_len)
        op_ids = self.rng.choice(self.augment_ids, size=self.n)
        for id in op_ids:
            op, minval, maxval = self.augment_list[id]
            val = float(self.m) * float(maxval - minval) + minval
            #print(val)
            img = op(img, val,random_state=self.rng,sfreq=self.sfreq,seq_len=seq_len)

        return img.permute(0,2,1).detach().view(max_seq_len,channel) #back to (len,channel)

class TransfromAugment:
    def __init__(self, names,m ,p=0.5,n=1, rd_seed=None,sfreq=100):
        print(f'Using Fix transfroms {names}, m={m}, n={n}, p={p}')
        self.p = p
        if isinstance(m,list):
            self.list_m = True
            assert len(m)==len(names)
            self.m_dic = {name:em for (name,em) in zip(names,m)}
        else:
            self.m_dic = {name:m for name in names}
        self.m = m      # [0, 1]
        self.n = n
        self.names = names
        self.rng = check_random_state(rd_seed)
        self.sfreq = sfreq
    def __call__(self, img, seq_len=None, **_kwargs): #ignore other args
        #print(img.shape)
        max_seq_len , channel = img.shape #(channel, seq_len)
        if seq_len==None:
            seq_len = max_seq_len
        img = img.permute(1,0).view(1,channel,max_seq_len) #(seq,ch)
        select_names = self.rng.choice(self.names, size=self.n)
        for name in select_names:
            augment = get_augment(name)
            use_op = self.rng.random() < self.p
            if use_op:
                op, minval, maxval = augment
                val = float(self.m_dic[name]) * float(maxval - minval) + minval
                img = op(img, val,random_state=self.rng,sfreq=self.sfreq,seq_len=seq_len)
            else: #pass
                pass
        return img.permute(0,2,1).detach().view(max_seq_len,channel) #back to (len,channel)

class TransfromAugment_classwise:
    def __init__(self, names,m ,p=0.5,n=1,num_class=None, rd_seed=None,sfreq=100,seq_len=None):
        print(f'Using Class-wise Fix transfroms {names}, m={m}, n={n}, p={p}')
        self.p = p
        assert len(m)==len(names)
        assert num_class==len(names)
        self.m_dic = {class_idx:(name.split('+'),em) for (class_idx,name,em) in zip(range(num_class),names,m)}
        print('Class-wise dic:', self.m_dic)
        self.m = m      # [0, 1]
        self.n = n
        self.names = names
        self.rng = check_random_state(rd_seed)
        self.sfreq = sfreq
        self.seq_len = seq_len
    def __call__(self, img, label):
        #print(img.shape)
        seq_len , channel = img.shape
        img = img.permute(1,0).view(1,channel,seq_len)
        #select_names = self.rng.choice(self.names, size=self.n)
        trans_name, mag = self.m_dic[label]
        select_names = self.rng.choice(trans_name, size=self.n)
        for name in select_names:
            augment = get_augment(name)
            use_op = self.rng.random() < self.p
            if use_op:
                op, minval, maxval = augment
                val = float(mag) * float(maxval - minval) + minval
                img = op(img, val,random_state=self.rng,sfreq=self.sfreq,seq_len=self.seq_len)
            else: #pass
                pass
        return img.permute(0,2,1).detach().view(seq_len,channel) #back to (len,channel)

class InfoRAugment:
    def __init__(self, names,m ,p=0.5,n=1,mode='a',sfreq=100,
        pw_len=0.2,qw_len=0.1,tw_len=0.4,rd_seed=None):
        print(f'Using Fix transfroms {names}, m={m}, n={n}, p={p}, mode={mode}')
        assert mode in ['a','p','qrs','t','n']
        self.detectors = Detectors(sfreq) #need input ecg: (seq_len)
        self.mode = mode
        self.sfreq = sfreq
        self.pw_len = pw_len
        self.qw_len = qw_len
        self.tw_len = tw_len
        if self.mode=='a':
            self.start_s,self.end_s = 0,None
        elif self.mode=='p':
            self.start_s,self.end_s = -0.2*sfreq,-0.06*sfreq
        elif self.mode=='qrs':
            self.start_s,self.end_s = -0.1*sfreq,-1
        elif self.mode=='t':
            self.start_s,self.end_s = 0.06*sfreq,0.4*sfreq
        else:
            self.start_s,self.end_s = 0.4*sfreq,-0.2*sfreq
        self.p = p
        if isinstance(m,list):
            self.list_m = True
            assert len(m)==len(names)
            self.m_dic = {name:em for (name,em) in zip(names,m)}
        else:
            self.m_dic = {name:m for name in names}
        self.m = m      # [0, 1]
        self.n = n
        self.names = names
        self.rng = check_random_state(rd_seed)
    def __call__(self, x):
        #print(img.shape)
        seq_len , channel = x.shape
        x = x.permute(1,0).view(1,channel,seq_len)
        select_lead = 0 #!!!tmp
        rpeaks_array = self.detectors.pan_tompkins_detector(x[0,select_lead,:])
        seg_list = [] #to segment
        start_point = 0 
        for end_point in rpeaks_array + [seq_len]:
            seg_list.append(x[:,:,start_point:end_point])
            start_point = end_point
        for i in range(1,len(seg_list)-1): #ignore first&last
            #calculate start and end
            seg_len = seg_list[i].shape[2]
            #print('seg len: ',seg_len)
            if self.end_s==None:
                seg_start,seg_end = 0,seg_len
            elif self.mode=='n' and seg_len<0.6*self.sfreq:
                seg_start,seg_end = 0,seg_len
            elif seg_len>0.4*self.sfreq:
                seg_start,seg_end = int(self.start_s),int(self.end_s)
                if seg_start<0:
                    seg_start = seg_len+seg_start
                if seg_end<0:
                    seg_end = seg_len+seg_end+1
                if seg_start>=seg_end:
                    seg_start = seg_end-1
            else: #heart beat too short
                seg_start,seg_end = 0,seg_len
            #augment
            select_names = self.rng.choice(self.names, size=self.n)
            for name in select_names:
                augment = get_augment(name)
                use_op = self.rng.random() < self.p
                if use_op:
                    op, minval, maxval = augment
                    val = float(self.m_dic[name]) * float(maxval - minval) + minval
                    seg_list[i] = op(seg_list[i], val,start=seg_start,end=seg_end,random_state=self.rng,sfreq=self.sfreq)
                else: #pass
                    pass
        new_x = torch.cat(seg_list,dim=2)
        return new_x.permute(0,2,1).detach().view(seq_len,channel) #back to (len,channel)

class BeatAugment:
    def __init__(self, names,m ,p=0.5,n=1,mode='a',
        sfreq=100,pw_len=0.2,qw_len=0.1,tw_len=0.4,
        reverse=False,rd_seed=None):
        print(f'Using Fix transfroms {names}, m={m}, n={n}, p={p}, mode={mode}')
        assert mode in ['a','b','p','t'] #a: all, b: heart beat(-0.2,0.4), p: p-wave(-0.2,0), t: t-wave(0,0.4)
        self.detectors = Detectors(sfreq) #need input ecg: (seq_len)
        self.mode = mode
        self.sfreq = sfreq
        self.pw_len = pw_len
        self.qw_len = qw_len
        self.tw_len = tw_len
        if self.mode=='a':
            self.start_s,self.end_s = 0,0
        elif self.mode=='p':
            self.start_s,self.end_s = -0.2*sfreq,0
        elif self.mode=='b':
            self.start_s,self.end_s = -0.2*sfreq,0.4*sfreq
        elif self.mode=='t':
            self.start_s,self.end_s = 0,0.4*sfreq
        self.reverse = reverse
        self.p = p
        if isinstance(m,list):
            self.list_m = True
            assert len(m)==len(names)
            self.m_dic = {name:em for (name,em) in zip(names,m)}
        else:
            self.m_dic = {name:m for name in names}
        self.m = m      # [0, 1]
        self.n = n
        self.names = names
        self.rng = check_random_state(rd_seed)
    def __call__(self, x):
        #print(img.shape)
        seq_len , channel = x.shape
        x = x.permute(1,0).view(1,channel,seq_len)
        select_lead = 0 #!!!tmp
        rpeaks_array = self.detectors.pan_tompkins_detector(x[0,select_lead,:])
        seg_list = [] #to segment
        start_point = 0
        for rpeak_point in rpeaks_array:
            beat_start = max(int(rpeak_point+self.start_s),start_point)
            beat_end = min(int(rpeak_point+self.end_s),seq_len)
            seg_list.append(x[:,:,start_point:beat_start])
            seg_list.append(x[:,:,beat_start:beat_end])
            start_point = beat_end
        #last
        seg_list.append(x[:,:,start_point:seq_len])
        #segment augment
        seg_start = 1
        seg_step = 2
        if self.mode=='a':
            seg_start = 0
            seg_step = 1
        elif self.reverse:
            seg_start = 0
        for i in range(seg_start,len(seg_list),seg_step):
            #calculate start and end
            seg_len = seg_list[i].shape[2]
            print('seg len: ',seg_len)
            if seg_len == 0:
                continue
            #augment
            select_names = self.rng.choice(self.names, size=self.n)
            for name in select_names:
                augment = get_augment(name)
                use_op = self.rng.random() < self.p
                if use_op:
                    op, minval, maxval = augment
                    val = float(self.m_dic[name]) * float(maxval - minval) + minval
                    seg_list[i] = op(seg_list[i], val,random_state=self.rng,sfreq=self.sfreq)
        new_x = torch.cat(seg_list,dim=2)
        assert new_x.shape[2]==seq_len
        return new_x.permute(0,2,1).detach().view(seq_len,channel) #back to (len,channel)

def stop_bn_track_running_stats(model):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.track_running_stats = False

def activate_bn_track_running_stats(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.track_running_stats = True

class KeepAugment(object): #need fix
    def __init__(self, mode, length,thres=0.6,transfrom=None,default_select=None, early=False, low = False,
        possible_segment=[1],grid_region=False, reverse=False,info_upper = 0.0,
        sfreq=100,pw_len=0.2,tw_len=0.4,**_kwargs):
        assert mode in ['auto','b','p','t','rand'] #auto: all, b: heart beat(-0.2,0.4), p: p-wave(-0.2,0), t: t-wave(0,0.4)
        self.mode = mode
        if self.mode=='p':
            self.start_s,self.end_s = -0.2*sfreq,0
        elif self.mode=='b':
            self.start_s,self.end_s = -0.2*sfreq,0.4*sfreq
        elif self.mode=='t':
            self.start_s,self.end_s = 0,0.4*sfreq
        self.length = length
        self.early = early
        self.low = low
        self.sfreq = sfreq #transfroms sfreq add for adaptive_augmentor
        self.pw_len = pw_len
        self.tw_len = tw_len
        self.trans = transfrom
        self.default_select = default_select
        self.thres = thres
        self.possible_segment = possible_segment
        self.grid_region = grid_region
        # normal, paste=> paste back important score higher then, cut=> not augment important region
        # when reverse, paste=> paste back important score lower then, cut=> augment important region
        self.reverse = reverse
        self.info_upper = info_upper
        self.detectors = Detectors(sfreq) #need input ecg: (seq_len)
        self.compare_func_list = [le,ge]
        #'torch.nn.functional.avg_pool1d' use this for segment
        ##self.m_pool = torch.nn.AvgPool1d(kernel_size=self.length, stride=1, padding=0) #for winodow sum
        print(f'Apply InfoKeep Augment: mode={self.mode}, threshold={self.thres}, transfrom={self.trans}')
    #func
    def get_augment(self,apply_func=None,selective='paste'):
        if apply_func!=None:
            augment = apply_func
        elif self.trans!=None:
            augment = self.trans
            if self.default_select:
                selective = self.default_select
        return augment, selective
    def get_selective(self,selective,thres=None,use_reverse=None):
        #cut or paste
        if thres==None:
            thres = self.thres
        if use_reverse==None:
            use_reverse = self.reverse
        assert selective in ['cut','paste']
        if selective=='cut':
            info_aug = thres
            com_idx = 0
            upper_bound = info_aug * self.info_upper
        else:
            info_aug = 1.0 - thres
            com_idx = 1
            upper_bound = 1.0 - (1.0 - info_aug) * self.info_upper
        if use_reverse:
            com_idx = (com_idx+1)%2
            upper_bound = 1.0 - upper_bound
        compare_func = self.compare_func_list[com_idx] #[lt, ge]
        bound_func = self.compare_func_list[(com_idx+1)%2] #[lt, ge]
        return info_aug, compare_func, upper_bound, bound_func
    def get_slc(self,t_series,model):
        t_series_ = t_series.clone().detach()
        if self.mode=='auto':
            t_series_.requires_grad = True
            slc_ = self.get_importance(model,t_series_)
        elif self.mode=='rand':
            slc_ = self.get_rand(t_series)
        else:
            slc_ = self.get_heartbeat(t_series)
        t_series_.requires_grad = False #no need gradient now
        return slc_, t_series_
    def get_seg(self,seg_number,seg_len,w,window_w,windowed_len):
        if self.grid_region:
            seg_accum = [i*seg_len for i in range(seg_number)]
            seg_accum.append(w)
            windowed_accum = [i*windowed_len for i in range(seg_number)]
            windowed_accum.append(window_w)
        else:
            seg_accum = [w for i in range(seg_number+1)]
            windowed_accum = [window_w for i in range(seg_number+1)]
        return seg_accum,windowed_accum
    #kwargs for apply_func, batch_inputs
    def __call__(self, t_series, model=None,selective='paste', apply_func=None, seq_len=None, **kwargs):
        b,w,c = t_series.shape
        augment, selective = self.get_augment(apply_func,selective)
        slc_, t_series_ = self.get_slc(t_series,model)
        info_aug, compare_func, info_bound, bound_func = self.get_selective(selective)
        #windowed_slc = self.m_pool(slc_.view(b,1,w)).view(b,-1)
        #select a segment number
        seg_number = np.random.choice(self.possible_segment)
        seg_len = int(w / seg_number)
        info_len = int(self.length/seg_number)
        windowed_slc = torch.nn.functional.avg_pool1d(slc_.view(b,1,w),kernel_size=info_len, stride=1, padding=0).view(b,-1)
        windowed_w = windowed_slc.shape[1]
        windowed_len = int(windowed_w / seg_number)
        #quant_scores = torch.quantile(windowed_slc,info_aug,dim=1) #quant for each batch
        seg_accum, windowed_accum = self.get_seg(seg_number,seg_len,w,windowed_w,windowed_len)
        #print(slc_)
        t_series_ = t_series_.detach().cpu()
        aug_t_s_list = []
        start, end = 0,w
        win_start, win_end = 0,windowed_w
        for i,(t_s, slc, windowed_slc_each, each_seq_len) in enumerate(zip(t_series_, slc_, windowed_slc, seq_len)):
            #find region for each segment
            region_list,inforegion_list = [],[]
            for seg_idx in range(seg_number):
                if self.grid_region:
                    start, end = seg_accum[seg_idx], seg_accum[seg_idx+1]
                    win_start, win_end = windowed_accum[seg_idx], windowed_accum[seg_idx+1]
                quant_score = torch.quantile(windowed_slc_each[win_start:win_end],info_aug)
                bound_score = torch.quantile(windowed_slc_each[win_start:win_end],info_bound)
                while(True):
                    x = np.random.randint(start,end)
                    x1 = np.clip(x - info_len // 2, 0, w)
                    x2 = np.clip(x + info_len // 2, 0, w)
                    reg_mean = slc[x1: x2].mean()
                    if compare_func(reg_mean,quant_score) and bound_func(reg_mean,bound_score):
                        region_list.append([x1,x2])
                        break
                info_region = t_s[x1: x2,:].clone().detach().cpu()
                inforegion_list.append(info_region)
            #augment & paste back
            if selective=='cut':
                t_s_aug = t_s.clone().detach().cpu()
                t_s_aug = augment(t_s_aug,i=i,seq_len=each_seq_len,**kwargs) #maybe some error!!!
                inforegion_list = [] #empty
                for (x1,x2) in region_list:
                    info_region = t_s_aug[x1: x2,:].clone().detach().cpu()
                    inforegion_list.append(info_region)
            else:
                t_s = augment(t_s,i=i,seq_len=each_seq_len,**kwargs) #some other augment if needed
                
            for reg_i in range(len(inforegion_list)):
                x1, x2 = region_list[reg_i][0], region_list[reg_i][1]
                t_s[x1: x2, :] = inforegion_list[reg_i]
            aug_t_s_list.append(t_s)
        #back
        if self.mode=='auto':
            model.train()
            for param in model.parameters():
                param.requires_grad = True
        return torch.stack(aug_t_s_list, dim=0) #(b,seq,ch)
    def Augment_search(self, t_series, model=None,selective='paste', apply_func=None,ops_names=None, seq_len=None, **kwargs):
        b,w,c = t_series.shape
        augment, selective = self.get_augment(apply_func,selective)
        slc_, t_series_ = self.get_slc(t_series,model)
        info_aug, compare_func, info_bound, bound_func = self.get_selective(selective)
        #windowed_slc = self.m_pool(slc_.view(b,1,w)).view(b,-1)
        #select a segment number
        seg_number = np.random.choice(self.possible_segment)
        seg_len = int(w / seg_number)
        info_len = int(self.length/seg_number)
        windowed_slc = torch.nn.functional.avg_pool1d(slc_.view(b,1,w),kernel_size=info_len, stride=1, padding=0).view(b,-1)
        windowed_w = windowed_slc.shape[1]
        windowed_len = int(windowed_w / seg_number)
        #quant_scores = torch.quantile(windowed_slc,info_aug,dim=1) #quant for each batch
        seg_accum, windowed_accum = self.get_seg(seg_number,seg_len,w,windowed_w,windowed_len)
        #print(slc_)
        #print(windowed_slc)
        #print(quant_scores)
        t_series_ = t_series_.detach().cpu()
        aug_t_s_list = []
        start, end = 0,w
        win_start, win_end = 0,windowed_w
        for i,(t_s, slc, windowed_slc_each, each_seq_len) in enumerate(zip(t_series_, slc_, windowed_slc,seq_len)):
            #find region
            for k, ops_name in enumerate(ops_names):
                t_s_tmp = t_s.clone().detach().cpu()
                region_list,inforegion_list = [],[]
                for seg_idx in range(seg_number):
                    if self.grid_region:
                        start, end = seg_accum[seg_idx], seg_accum[seg_idx+1]
                        win_start, win_end = windowed_accum[seg_idx], windowed_accum[seg_idx+1]
                    quant_score = torch.quantile(windowed_slc_each[win_start: win_end],info_aug)
                    bound_score = torch.quantile(windowed_slc_each[win_start:win_end],info_bound)
                    while(True):
                        x = np.random.randint(start,end)
                        x1 = np.clip(x - info_len // 2, 0, w)
                        x2 = np.clip(x + info_len // 2, 0, w)
                        reg_mean = slc[x1: x2].mean()
                        if compare_func(reg_mean,quant_score) and bound_func(reg_mean,bound_score):
                            region_list.append([x1,x2])
                            break
                    info_region = t_s_tmp[x1: x2,:].clone().detach().cpu()
                    inforegion_list.append(info_region)
                #augment & paste back
                if selective=='cut':
                    t_s_aug = t_s_tmp.clone().detach().cpu()
                    t_s_aug = augment(t_s_aug,i=i,k=k,ops_name=ops_name,seq_len=each_seq_len,**kwargs) #maybe some error!!!
                    inforegion_list = [] #empty
                    for (x1,x2) in region_list:
                        info_region = t_s_aug[x1: x2,:].clone().detach().cpu()
                        inforegion_list.append(info_region)
                else:
                    t_s_tmp = augment(t_s_tmp,i=i,k=k,ops_name=ops_name,seq_len=each_seq_len,**kwargs) #some other augment if needed
                    #print('Size compare: ',t_s[x1: x2, :].shape,info_region.shape)
                for reg_i in range(len(inforegion_list)):
                    x1, x2 = region_list[reg_i][0], region_list[reg_i][1]
                    t_s_tmp[x1: x2, :] = inforegion_list[reg_i]
                aug_t_s_list.append(t_s_tmp)
        #back
        if self.mode=='auto':
            model.train()
            for param in model.parameters():
                param.requires_grad = True
        return torch.stack(aug_t_s_list, dim=0) #(b*ops,seq,ch)
    def get_importance(self, model, x, **_kwargs):
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, 'lstm'):
            model.lstm.train()
            stop_bn_track_running_stats(model)
        else:
            model.eval()
        b, seq_len , c = x.shape
        if self.early: #early not allow now
            preds = model(x,early=True)
        else:
            preds = model(x)

        score, _ = torch.max(preds, 1) #predict class
        score.mean().backward() #among batch mean
        slc_, _ = torch.max(torch.abs(x.grad), dim=2) #max of channel
        
        b,w = slc_.shape #1d only
        slc_ = slc_.view(slc_.size(0), -1)
        slc_ -= slc_.min(1, keepdim=True)[0]
        slc_ /= slc_.max(1, keepdim=True)[0]
        slc_ = slc_.view(b, w)
        if hasattr(model, 'lstm'):
            activate_bn_track_running_stats(model)
        return slc_   
    def get_heartbeat(self,x):
        b, seq_len , channel = x.shape
        imp_map_list = []
        for x_each in x:
            select_lead = 0 #!!!tmp
            rpeaks_array = self.detectors.pan_tompkins_detector(x_each[:,select_lead])
            imp_map = np.zeros((seq_len,), np.float32) #maybe need smooth!!!
            for rpeak_point in rpeaks_array:
                r1 = int(np.clip(rpeak_point + self.start_s, 0, seq_len))
                r2 = int(np.clip(rpeak_point + self.end_s, 0, seq_len))
                imp_map[r1:r2] += 1
            #normalize to mean of all sequence=1
            ratio = seq_len / np.sum(imp_map)
            imp_map *= ratio
            imp_map = torch.from_numpy(imp_map)
            imp_map_list.append(imp_map)
        return torch.stack(imp_map_list, dim=0) #(b,seq)
    def get_rand(self,x):
        b, seq_len , channel = x.shape
        imp_map_list = []
        for x_each in x:
            imp_map = torch.rand(seq_len)
            imp_map_list.append(imp_map)
        return torch.stack(imp_map_list, dim=0) #(b,seq)
#segment gradient
def stop_gradient_keep(trans_image, magnitude, keep_thre, region_list):
    x1, x2 = region_list[0][0], region_list[0][1]
    images = trans_image #(seq, ch)
    adds = 0
    images = images - magnitude
    adds = adds + magnitude
    for (x1,x2) in region_list:
        info_part = images[x1:x2,:]
        info_part = info_part - keep_thre
    #add gradient
    images = images.detach() + adds
    for (x1,x2) in region_list:
        info_part = images[x1:x2,:]
        info_part = info_part + keep_thre
    return images
class AdaKeepAugment(KeepAugment): #
    def __init__(self, mode, length,thres=0.6,transfrom=None,default_select=None, early=False, low = False,
        possible_segment=[1],grid_region=False, reverse=False,info_upper = 0.0, thres_adapt=True, adapt_target='len',
        sfreq=100,pw_len=0.2,tw_len=0.4,**_kwargs):
        assert mode in ['auto','b','p','t','rand'] #auto: all, b: heart beat(-0.2,0.4), p: p-wave(-0.2,0), t: t-wave(0,0.4)
        self.mode = mode
        if self.mode=='p':
            self.start_s,self.end_s = -0.2*sfreq,0
        elif self.mode=='b':
            self.start_s,self.end_s = -0.2*sfreq,0.4*sfreq
        elif self.mode=='t':
            self.start_s,self.end_s = 0,0.4*sfreq
        assert adapt_target in ['len','seg','way']
        self.adapt_target = adapt_target
        self.way = [('cut',False),('cut',True),('paste',False),('paste',True)] #(selective,reverse)
        self.length = length #len is a list if adapt target 
        self.early = early
        self.low = low
        self.sfreq = sfreq #transfroms sfreq add for adaptive_augmentor
        self.pw_len = pw_len
        self.tw_len = tw_len
        self.trans = transfrom
        self.default_select = default_select
        self.thres = thres
        self.thres_adapt=thres_adapt
        self.possible_segment = possible_segment
        self.grid_region = grid_region
        self.reverse = reverse
        self.info_upper = info_upper
        self.detectors = Detectors(sfreq) #need input ecg: (seq_len)
        self.compare_func_list = [le,ge]
        self.all_stages = ['trans','keep']
        self.stage = 0
        #'torch.nn.functional.avg_pool1d' use this for segment
        print(f'Apply InfoKeep Augment: mode={self.mode},target={self.adapt_target}, threshold={self.thres}, transfrom={self.trans}')
    #kwargs for apply_func, batch_inputs
    def __call__(self, t_series, model=None,selective='paste', apply_func=None,len_idx=None, keep_thres=None, seq_len=None, **kwargs):
        b,w,c = t_series.shape
        augment, selective = self.get_augment(apply_func,selective)
        slc_, t_series_ = self.get_slc(t_series,model)
        #windowed_slc = self.m_pool(slc_.view(b,1,w)).view(b,-1)
        #select a segment number
        seg_number = np.random.choice(self.possible_segment)
        total_len = self.length[0]
        #print(slc_)
        t_series_ = t_series_.detach().cpu()
        aug_t_s_list = []
        start, end = 0,w
        for i,(t_s, slc,each_seq_len) in enumerate(zip(t_series_, slc_,seq_len)):
            #len choose
            use_reverse = None
            if self.adapt_target=='len':
                total_len = self.length[len_idx[i]]
            elif self.adapt_target=='way':
                select_way = self.way[len_idx[i]]
                selective = select_way[0]
                use_reverse = select_way[1]
            elif self.adapt_target=='seg':
                seg_number = self.possible_segment[len_idx[i]]
            else:
                raise 
            info_aug, compare_func, info_bound, bound_func = self.get_selective(selective,thres=keep_thres[i],use_reverse=use_reverse)
            info_len = int(total_len/seg_number)
            windowed_slc = torch.nn.functional.avg_pool1d(slc.view(1,1,w),kernel_size=info_len, stride=1, padding=0).view(1,-1)
            windowed_w = windowed_slc.shape[1]
            windowed_len = int(windowed_w / seg_number)
            seg_len = int(w / seg_number)
            seg_accum, windowed_accum = self.get_seg(seg_number,seg_len,w,windowed_w,windowed_len)
            windowed_slc_each = windowed_slc[0]
            win_start, win_end = 0,windowed_w
            #find region for each segment
            region_list,inforegion_list = [],[]
            for seg_idx in range(seg_number):
                if self.grid_region:
                    #start, end = seg_accum[seg_idx], seg_accum[seg_idx+1] #calculate all window no need this
                    win_start, win_end = windowed_accum[seg_idx], windowed_accum[seg_idx+1]
                seg_window = windowed_slc_each[win_start:win_end]
                quant_score = torch.quantile(seg_window,info_aug)
                bound_score = torch.quantile(seg_window,info_bound)
                select_windows = (compare_func(seg_window,quant_score) & bound_func(seg_window,bound_score)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
                if len(select_windows)==0:
                    print('origin window: ', seg_window)
                    print('window index:', select_windows.shape)
                    print('no result:')
                    print('quant_score: ',quant_score)
                    print('bound_score: ',bound_score)
                    print('max windows: ',torch.max(seg_window))
                select_p = np.random.choice(select_windows) + win_start #window start for adjust
                x = select_p + info_len // 2 #back to seg
                x1 = np.clip(x - info_len // 2, 0, w)
                x2 = np.clip(x + info_len // 2, 0, w)
                region_list.append([x1,x2])
                '''while(True):
                    x = np.random.randint(start,end)
                    x1 = np.clip(x - info_len // 2, 0, w)
                    x2 = np.clip(x + info_len // 2, 0, w)
                    reg_mean = slc[x1: x2].mean()
                    if compare_func(reg_mean,quant_score) and bound_func(reg_mean,bound_score):
                        region_list.append([x1,x2])
                        break'''
                info_region = t_s[x1: x2,:].clone().detach().cpu()
                inforegion_list.append(info_region)
            #augment & paste back
            if selective=='cut':
                t_s_aug = t_s.clone().detach().cpu()
                t_s_aug = augment(t_s_aug,i=i,seq_len=each_seq_len,**kwargs) #maybe some error!!!
                inforegion_list = [] #empty
                for (x1,x2) in region_list:
                    info_region = t_s_aug[x1: x2,:].clone().detach().cpu()
                    inforegion_list.append(info_region)
            else:
                t_s = augment(t_s,i=i,seq_len=each_seq_len,**kwargs) #some other augment if needed
            #paste back
            for reg_i in range(len(inforegion_list)):
                x1, x2 = region_list[reg_i][0], region_list[reg_i][1]
                t_s[x1: x2, :] = inforegion_list[reg_i]
            aug_t_s_list.append(t_s)
        #back
        if self.mode=='auto':
            model.train()
            for param in model.parameters():
                param.requires_grad = True
        return torch.stack(aug_t_s_list, dim=0) #(b,seq,ch)
    def make_params(self,adapt_target,each_len=None,seg_number=None,selective=None):
        if adapt_target=='len': #search over keep len or keep segment
            keepway_params = [(selective,self.reverse) for i in range(len(self.length))]
            keeplen_params = self.length
            keepseg_params = [seg_number for i in range(len(self.length))]
        elif adapt_target=='way':
            keepway_params = self.way
            keeplen_params = [each_len for i in range(len(self.way))]
            keepseg_params = [seg_number for i in range(len(self.way))]
        elif adapt_target=='seg':
            keepway_params = [(selective,self.reverse) for i in range(len(self.possible_segment))]
            keeplen_params = [each_len for i in range(len(self.possible_segment))]
            keepseg_params = self.possible_segment
        else:
            raise

        return keepway_params, keeplen_params, keepseg_params
    def Augment_search(self, t_series, model=None,selective='paste', apply_func=None,ops_names=None, keep_thres=None, seq_len=None, **kwargs):
        b,w,c = t_series.shape
        augment, selective = self.get_augment(apply_func,selective)
        slc_, t_series_ = self.get_slc(t_series,model)
        magnitudes = kwargs['magnitudes']
        t_series_ = t_series_.detach().cpu()
        aug_t_s_list = []
        each_len, seg_number = self.length[0], self.possible_segment[0]
        keepway_params, keeplen_params, keepseg_params = self.make_params(self.adapt_target,each_len,seg_number,selective)
        for i,(t_s, slc,each_seq_len) in enumerate(zip(t_series_, slc_,seq_len)):
            for (each_way,each_len, seg_number) in zip(keepway_params,keeplen_params,keepseg_params):

                (selective, use_reverse) = each_way
                info_aug, compare_func, info_bound, bound_func = self.get_selective(selective,thres=keep_thres[i],use_reverse=use_reverse)
                #select a segment number
                info_len = int(each_len/seg_number)
                windowed_slc = torch.nn.functional.avg_pool1d(slc.view(1,1,w),kernel_size=info_len, stride=1, padding=0).view(1,-1)
                windowed_w = windowed_slc.shape[1]
                windowed_len = int(windowed_w / seg_number)
                seg_len = int(w / seg_number)
                seg_accum, windowed_accum = self.get_seg(seg_number,seg_len,w,windowed_w,windowed_len)
                windowed_slc_each = windowed_slc[0]
                win_start, win_end = 0,windowed_w
                #find region
                for k, ops_name in enumerate(ops_names):
                    t_s_tmp = t_s.clone().detach().cpu()
                    region_list,inforegion_list = [],[]
                    for seg_idx in range(seg_number):
                        if self.grid_region:
                            start, end = seg_accum[seg_idx], seg_accum[seg_idx+1]
                            win_start, win_end = windowed_accum[seg_idx], windowed_accum[seg_idx+1]
                        seg_window = windowed_slc_each[win_start:win_end]
                        quant_score = torch.quantile(seg_window,info_aug)
                        bound_score = torch.quantile(seg_window,info_bound)
                        select_windows = (compare_func(seg_window,quant_score) & bound_func(seg_window,bound_score)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
                        if len(select_windows)==0:
                            print('origin window: ', seg_window)
                            print('window index', select_windows.shape)
                            print('no result:')
                            print('quant_score: ',quant_score)
                            print('bound_score: ',bound_score)
                            print('max windows: ',torch.max(seg_window))
                        select_p = np.random.choice(select_windows) + win_start #window start for adjust
                        x = select_p + info_len // 2 #back to seg
                        x1 = np.clip(x - info_len // 2, 0, w)
                        x2 = np.clip(x + info_len // 2, 0, w)
                        region_list.append([x1,x2])
                        info_region = t_s_tmp[x1: x2,:].clone().detach().cpu()
                        inforegion_list.append(info_region)
                    #augment & paste back
                    if selective=='cut':
                        t_s_aug = t_s_tmp.clone().detach().cpu()
                        t_s_aug = augment(t_s_aug,i=i,k=k,ops_name=ops_name,keep_thres=keep_thres,seq_len=each_seq_len,**kwargs) #maybe some error!!!
                        inforegion_list = [] #empty
                        for (x1,x2) in region_list:
                            info_region = t_s_aug[x1: x2,:].clone().detach().cpu()
                            inforegion_list.append(info_region)
                        #for info_region in inforegion_list:
                        #    info_region = augment(info_region,i=i,k=k,ops_name=ops_name,keep_thres=keep_thres,**kwargs)
                    else:
                        t_s_tmp = augment(t_s_tmp,i=i,k=k,ops_name=ops_name,keep_thres=keep_thres,seq_len=each_seq_len,**kwargs) #some other augment if needed
                        #print('Size compare: ',t_s[x1: x2, :].shape,info_region.shape)
                    for reg_i in range(len(inforegion_list)):
                        x1, x2 = region_list[reg_i][0], region_list[reg_i][1]
                        t_s_tmp[x1: x2, :] = inforegion_list[reg_i]
                    t_s_tmp = stop_gradient_keep(t_s_tmp.cuda(), magnitudes[i][k], keep_thres[i],region_list) #add keep thres
                    aug_t_s_list.append(t_s_tmp)
        #back
        if self.mode=='auto':
            model.train()
            for param in model.parameters():
                param.requires_grad = True
        return torch.stack(aug_t_s_list, dim=0) #(b*lens*ops,seq,ch)
    
    #independent search, ops_names is this turn params and fix_idx is this turn fixs
    ###!!!Not a good implement!!!###
    def Augment_search_ind(self, t_series, model=None,selective='paste', apply_func=None,ops_names=None,fix_idx=None, keep_thres=None,seq_len=None, **kwargs):
        b,w,c = t_series.shape
        augment, selective = self.get_augment(apply_func,selective)
        slc_, t_series_ = self.get_slc(t_series,model)
        magnitudes = kwargs['magnitudes']
        t_series_ = t_series_.detach().cpu()
        aug_t_s_list = []
        stage_name = self.all_stages[self.stage]
        #keep param or transfrom param
        if stage_name=='trans':
            keeplen_params = fix_idx
        #keep len or segment
        each_len, seg_number = self.length[0], self.possible_segment[0]
        keepway_params, keeplen_params, keepseg_params = self.make_params(self.adapt_target,each_len,seg_number,selective)
        for i,(t_s, slc,each_seq_len) in enumerate(zip(t_series_, slc_,seq_len)):
            if stage_name=='trans': #from all possible to a fix number
                keepway_params_l = [keepway_params[fix_idx[i]]]
                keeplen_params_l = [keeplen_params[fix_idx[i]]]
                keepseg_params_l = [keepseg_params[fix_idx[i]]]
            else:
                keepway_params_l,keeplen_params_l,keepseg_params_l = keepway_params,keeplen_params,keepseg_params
            for (each_way,each_len, seg_number) in zip(keepway_params_l,keeplen_params_l,keepseg_params_l):
                (selective, use_reverse) = each_way
                info_aug, compare_func, info_bound, bound_func = self.get_selective(selective,thres=keep_thres[i],use_reverse=use_reverse)
                #select a segment number
                info_len = int(each_len/seg_number)
                windowed_slc = torch.nn.functional.avg_pool1d(slc.view(1,1,w),kernel_size=info_len, stride=1, padding=0).view(1,-1)
                windowed_w = windowed_slc.shape[1]
                windowed_len = int(windowed_w / seg_number)
                seg_len = int(w / seg_number)
                seg_accum, windowed_accum = self.get_seg(seg_number,seg_len,w,windowed_w,windowed_len)
                windowed_slc_each = windowed_slc[0]
                win_start, win_end = 0,windowed_w
                #find region
                if stage_name=='keep': #from all possible to a fix number
                    ops_names_l = [ops_names[fix_idx[i]]]
                else:
                    ops_names_l = ops_names
                for k, ops_name in enumerate(ops_names_l):
                    t_s_tmp = t_s.clone().detach().cpu()
                    region_list,inforegion_list = [],[]
                    for seg_idx in range(seg_number):
                        if self.grid_region:
                            start, end = seg_accum[seg_idx], seg_accum[seg_idx+1]
                            win_start, win_end = windowed_accum[seg_idx], windowed_accum[seg_idx+1]
                        seg_window = windowed_slc_each[win_start:win_end]
                        quant_score = torch.quantile(seg_window,info_aug)
                        bound_score = torch.quantile(seg_window,info_bound)
                        select_windows = (compare_func(seg_window,quant_score) & bound_func(seg_window,bound_score)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
                        if len(select_windows)==0:
                            print('origin window: ', seg_window)
                            print('window index', select_windows.shape)
                            print('no result:')
                            print('quant_score: ',quant_score)
                            print('bound_score: ',bound_score)
                            print('max windows: ',torch.max(seg_window))
                        select_p = np.random.choice(select_windows) + win_start #window start for adjust
                        x = select_p + info_len // 2 #back to seg
                        x1 = np.clip(x - info_len // 2, 0, w)
                        x2 = np.clip(x + info_len // 2, 0, w)
                        region_list.append([x1,x2])
                        info_region = t_s_tmp[x1: x2,:].clone().detach().cpu()
                        inforegion_list.append(info_region)
                    #augment & paste back
                    if selective=='cut':
                        t_s_aug = t_s_tmp.clone().detach().cpu()
                        t_s_aug = augment(t_s_aug,i=i,k=k,ops_name=ops_name,keep_thres=keep_thres,seq_len=each_seq_len,**kwargs) #maybe some error!!!
                        inforegion_list = [] #empty
                        for (x1,x2) in region_list:
                            info_region = t_s_aug[x1: x2,:].clone().detach().cpu()
                            inforegion_list.append(info_region)
                    else:
                        t_s_tmp = augment(t_s_tmp,i=i,k=k,ops_name=ops_name,keep_thres=keep_thres,seq_len=each_seq_len,**kwargs) #some other augment if needed
                        #print('Size compare: ',t_s[x1: x2, :].shape,info_region.shape)
                    for reg_i in range(len(inforegion_list)):
                        x1, x2 = region_list[reg_i][0], region_list[reg_i][1]
                        t_s_tmp[x1: x2, :] = inforegion_list[reg_i]
                    t_s_tmp = stop_gradient_keep(t_s_tmp.cuda(), magnitudes[i][k], keep_thres[i],region_list) #add keep thres
                    aug_t_s_list.append(t_s_tmp)
        #back
        if self.mode=='auto':
            model.train()
            for param in model.parameters():
                param.requires_grad = True
        out_ts = torch.stack(aug_t_s_list, dim=0)
        return out_ts #(b*lens,seq,ch) or (b*ops,seq,ch)
    
    def change_stage(self):
        #stage change
        self.stage = (self.stage+1) % len(self.all_stages)

if __name__ == '__main__':
    print('Test all operations')
    from datasets import EDFX,PTBXL,Chapman,WISDM
    #t = np.linspace(0, 10, 1000)
    #x = np.vstack([np.cos(t),np.sin(t),np.random.normal(0, 0.3, 1000)]).T
    Freq_dict = {
    'edfx' : 100,
    'ptbxl' : 100,
    'wisdm' : 20,
    'chapman' : 500,
    }
    TimeS_dict = {
    'edfx' : 30,
    'ptbxl' : 10,
    'wisdm' : 10,
    'chapman' : 10,
    }
    #dataset = PTBXL(dataset_path='../CWDA_research/CWDA/datasets/Datasets/ptbxl-dataset',mode='test',labelgroup='superdiagnostic',denoise=True)
    dataset = PTBXL(dataset_path='../CWDA_research/CWDA/datasets/Datasets/ptbxl-dataset',mode='test',labelgroup='superdiagnostic')
    print(dataset[0])
    print(dataset[0][0].shape)
    sample = dataset[100]
    x = sample[0]
    t = np.linspace(0, TimeS_dict['ptbxl'], 1000)
    label = sample[2]
    print(t.shape)
    print(x.shape)
    x_tensor = torch.from_numpy(x).float()
    test_ops = EXP_TEST_NAMES
    #
    plot_line(t,x,title='identity')
    for name in test_ops:
        for m in [0,0.1,0.5,0.98]:
            trans_aug = TransfromAugment([name],m=m,n=1,p=0.5)
            x_aug = trans_aug(x_tensor).numpy()
            print(x_aug.shape)
            plot_line(t,x_aug,f'{name}_m:{m}')
    #beat aug
    '''plot_line(t,x,title='identity')
    for each_mode in ['b','p','t']:
        for name in test_ops:
            for m in [0,0.1,0.5,0.98]:
                print(each_mode,'='*10,name,'='*10,m)
                info_aug = BeatAugment([name],m=m,p=1.0,mode=each_mode)
                x_aug = info_aug(x_tensor).numpy()
                print(x_aug.shape)
                plot_line(t,x_aug,f'{name}_mode:{each_mode}_m:{m}')'''
    #keep aug
    '''plot_line(t,x,title='identity')
    x_tensor = torch.unsqueeze(x_tensor,dim=0)
    for each_mode in ['b','p','t']:
        for name in test_ops:
            for m in [0.5,0.98]:
                print(each_mode,'='*10,name,'='*10,m)
                info_aug = KeepAugment(transfrom=TransfromAugment([name],m=m,p=1.0),mode=each_mode,length=200,default_select='paste')
                print(x_tensor.shape)
                x_aug = info_aug(x_tensor)
                print(x_tensor.shape)
                x_aug = torch.squeeze(x_aug,dim=0).numpy()
                print(x_aug.shape)
                plot_line(t,x_aug,f'{name}_mode:{each_mode}_m:{m}')'''
    #randaug
    '''randaug = RandAugment(1,0,rd_seed=42)
    name = 'random_time_mask'
    for i in range(3):
        #print('='*10,name,'='*10)
        x_aug = randaug(x_tensor).numpy()
        print(x_aug.shape)
        plot_line(t,x_aug)'''
    #ECG part
    '''print('ECG Augmentation')
    for name in ECG_OPS_NAMES:
        print('='*10,name,'='*10)
        x_aug = apply_augment(x_tensor,name,0.5).numpy()
        print(x_aug.shape)
        plot_line(t,x_aug)'''
    '''name = 'QRS_resample'
    for i in range(3):
        print('='*10,name,'='*10)
        x_aug = apply_augment(x_tensor,name,0.5,rd_seed=None).numpy()
        print(x_aug.shape)
        plot_line(t,x_aug)'''
