# code is adapted from CADDA and braincode
from numbers import Real
import random
import numpy as np
import torch
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
#for this file (channel, len) !!!
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

def permute_channels(X, permutation, *args, **kwargs):
    return X[..., permutation, :]


def _sample_mask_start(X, mask_len_samples, random_state):
    rng = check_random_state(random_state)
    seq_length = torch.as_tensor(X.shape[-1], device=X.device)
    mask_start = torch.as_tensor(rng.uniform(
        low=0, high=1, size=X.shape[0],
    ), device=X.device) * (seq_length - mask_len_samples)
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
def random_time_mask(X, mask_len_samples, random_state=None, *args, **kwargs):
    mask_start = _sample_mask_start(X, mask_len_samples, random_state)
    return _relaxed_mask_time(X, mask_start, mask_len_samples)
def exp_time_mask(X, mask_len_samples, random_state=None, *args, **kwargs):
    seq_len = X.shape[2]
    all_mask_len_samples = int(seq_len * mask_len_samples / 100.0)
    mask_start = _sample_mask_start(X, all_mask_len_samples, random_state)
    return _mask_time(X, mask_start, all_mask_len_samples)
def _sample_mask_start_info(X, mask_len_samples,start,end, random_state):
    rng = check_random_state(random_state)
    seq_length = X.shape[-1]
    c_start = max(min(start,end-mask_len_samples),0)
    c_end = min(max(end,c_start+mask_len_samples),seq_length-mask_len_samples)
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
    starts = rng.randint(low=start, high=max(end,start+1), size=(x.shape[0])).astype(int)
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

def Window_Slicing_Circle(X, magnitude,window_size=1000, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
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

def window_warp(x,rng, window_ratio=0.1, scales=[0.5, 2.],start=0,end=None): #ref (batch, time_steps, channel)
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
    window_starts = rng.randint(low=start+1, high=max(end,2), size=(x.shape[0])).astype(int)
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
        (random_time_mask, 0, 100),  # 5 impl
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
]
EXP_TEST_NAMES =[
    'exp_time_mask',
    'exp_bandstop',
    'exp_freq_shift',
]
INFO_EXP_LIST = [
    (info_time_mask, 0, 100),
    (info_Window_Warp, 0, 1), #sample freq=100, bandstop=48 because of notch
    (info_Window_Slicing, 0, 1), #sample freq=100
]
INFO_TEST_NAMES =[
    'info_time_mask',
    'info_Window_Warp',
    'info_Window_Slicing',
]

AUGMENT_DICT = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in TS_AUGMENT_LIST+ECG_AUGMENT_LIST+TS_ADD_LIST+TS_EXP_LIST+INFO_EXP_LIST}
def get_augment(name):
    return AUGMENT_DICT[name]

def apply_augment(img, name, level, rd_seed=None):
    augment_fn, low, high = get_augment(name)
    assert 0 <= level
    assert level <= 1
    #change tseries signal from (len,channel) to (batch,channel,len)
    #print('Device: ',img.device)
    seq_len , channel = img.shape
    img = img.permute(1,0).view(1,channel,seq_len)
    aug_value = level * (high - low) + low
    #print('Device: ',aug_value.device)
    aug_img = augment_fn(img, aug_value,random_state=rd_seed)
    return aug_img.permute(0,2,1).detach().view(seq_len,channel) #back to (len,channel)

def plot_line(t,x):
    plt.clf()
    channel_num = x.shape[-1]
    for i in  range(channel_num):
        plt.plot(t, x[:,i])
    plt.show()

class ToTensor:
    def __init__(self) -> None:
        pass
    def __call__(self, img):
        return torch.tensor(img).float()

class RandAugment:
    def __init__(self, n, m, rd_seed=None,augselect=''):
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
    def __call__(self, img):
        #print(img.shape)
        seq_len , channel = img.shape
        img = img.permute(1,0).view(1,channel,seq_len)
        op_ids = self.rng.choice(self.augment_ids, size=self.n)
        for id in op_ids:
            op, minval, maxval = self.augment_list[id]
            val = float(self.m) * float(maxval - minval) + minval
            #print(val)
            img = op(img, val,random_state=self.rng)

        return img.permute(0,2,1).detach().view(seq_len,channel) #back to (len,channel)

class TransfromAugment:
    def __init__(self, names,m ,p=0.5,n=1, rd_seed=None):
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
    def __call__(self, img):
        #print(img.shape)
        seq_len , channel = img.shape
        img = img.permute(1,0).view(1,channel,seq_len)
        select_names = self.rng.choice(self.names, size=self.n)
        for name in select_names:
            augment = get_augment(name)
            use_op = self.rng.random() < self.p
            if use_op:
                op, minval, maxval = augment
                val = float(self.m_dic[name]) * float(maxval - minval) + minval
                img = op(img, val,random_state=self.rng)
            else: #pass
                pass
        return img.permute(0,2,1).detach().view(seq_len,channel) #back to (len,channel)

class TransfromAugment_classwise:
    def __init__(self, names,m ,p=0.5,n=1,num_class=None, rd_seed=None):
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
                img = op(img, val,random_state=self.rng)
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
            self.start_s,self.end_s = -0.2*sfreq,-1
        elif self.mode=='qrs':
            self.start_s,self.end_s = -0.1*sfreq,-1
        elif self.mode=='t':
            self.start_s,self.end_s = 0,0.4*sfreq
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
            if self.end_s==None:
                seg_start,seg_end = 0,seg_len
            elif self.mode=='n' and seg_len<0.6*self.sfreq:
                seg_start,seg_end = 0,seg_len
            elif seg_len>0.4*self.sfreq:
                seg_start,seg_end = int(self.start_s),int(self.end_s)
                if seg_start<0:
                    seg_start = seg_len-seg_start
                if seg_end<0:
                    seg_end = seg_len-seg_end+1
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
                    seg_list[i] = op(seg_list[i], val,seg_start,seg_end,random_state=self.rng)
                else: #pass
                    pass
        new_x = torch.cat(seg_list,dim=2)
        return new_x.permute(0,2,1).detach().view(seq_len,channel) #back to (len,channel)

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
    plot_line(t,x)
    for name in EXP_TEST_NAMES:
        print('='*10,name,'='*10)
        x_aug = apply_augment(x_tensor,name,0.98).numpy()
        print(x_aug.shape)
        plot_line(t,x_aug)
    randaug = RandAugment(1,0,rd_seed=42)
    name = 'random_time_mask'
    for i in range(3):
        #print('='*10,name,'='*10)
        x_aug = randaug(x_tensor).numpy()
        print(x_aug.shape)
        plot_line(t,x_aug)
    #ECG part
    '''print('ECG Augmentation')
    for name in ECG_OPS_NAMES:
        print('='*10,name,'='*10)
        x_aug = apply_augment(x_tensor,name,0.5).numpy()
        print(x_aug.shape)
        plot_line(t,x_aug)'''
    name = 'QRS_resample'
    for i in range(3):
        print('='*10,name,'='*10)
        x_aug = apply_augment(x_tensor,name,0.5,rd_seed=None).numpy()
        print(x_aug.shape)
        plot_line(t,x_aug)
