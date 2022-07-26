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
        mask[i, :, start:start + mask_len_samples] = 0
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


def _torch_normalize_vectors(rr):
    """Normalize surface vertices."""
    new_rr = rr.clone()
    size = torch.linalg.norm(rr, axis=1)
    mask = (size > 0)
    if not torch.all(mask):
        new_rr[mask] = rr[mask] / size[mask].unsqueeze(-1)
    else:
        # For some reason, this distinction is necessary when the input (rr)
        # depends on learnable parameters. Passing a mask made of all True
        # will lead to a RuntimeError: number of dims don't match in permute
        new_rr = rr / size.unsqueeze(-1)
    return new_rr

def _torch_legval(x, c, tensor=True):
    """
    Evaluate a Legendre series at points x.
    If `c` is of length `n + 1`, this function returns the value:
    .. math:: p(x) = c_0 * L_0(x) + c_1 * L_1(x) + ... + c_n * L_n(x)
    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `c`.
    If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If
    `c` is multidimensional, then the shape of the result depends on the
    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
    scalars have shape (,).
    Trailing zeros in the coefficients will be used in the evaluation, so
    they should be avoided if efficiency is a concern.
    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        with themselves and with the elements of `c`.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n]. If `c` is multidimensional the
        remaining indices enumerate multiple polynomials. In the two
        dimensional case the coefficients may be thought of as stored in
        the columns of `c`.
    tensor : boolean, optional
        If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`. Scalars have dimension 0
        for this action. The result is that every column of coefficients in
        `c` is evaluated for every element of `x`. If False, `x` is broadcast
        over the columns of `c` for the evaluation.  This keyword is useful
        when `c` is multidimensional. The default value is True.
        .. versionadded:: 1.7.0
    Returns
    -------
    values : ndarray, algebra_like
        The shape of the return value is described above.
    See Also
    --------
    legval2d, leggrid2d, legval3d, leggrid3d
    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.
    Examples
    --------
    """
    c = torch.as_tensor(c)
    c = c.double()
    if isinstance(x, (tuple, list)):
        x = torch.as_tensor(x)
    if isinstance(x, torch.Tensor) and tensor:
        c = c.view(c.shape + (1,) * x.ndim)

    c = c.to(x.device)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * x * (2 * nd - 1)) / nd
    return c0 + c1 * x


def _torch_calc_g(cosang, stiffness=4, n_legendre_terms=50):
    """Calculate spherical spline g function between points on a sphere.
    Parameters
    ----------
    cosang : array-like of float, shape(n_channels, n_channels)
        cosine of angles between pairs of points on a spherical surface. This
        is equivalent to the dot product of unit vectors.
    stiffness : float
        stiffness of the spline.
    n_legendre_terms : int
        number of Legendre terms to evaluate.
    Returns
    -------
    G : np.ndrarray of float, shape(n_channels, n_channels)
        The G matrix.
    """
    factors = [(2 * n + 1) / (n ** stiffness * (n + 1) ** stiffness *
                              4 * np.pi)
               for n in range(1, n_legendre_terms + 1)]
    return _torch_legval(cosang, [0] + factors)


def _torch_make_interpolation_matrix(pos_from, pos_to, alpha=1e-5):
    """Compute interpolation matrix based on spherical splines.
    Implementation based on [1]
    Parameters
    ----------
    pos_from : np.ndarray of float, shape(n_good_sensors, 3)
        The positions to interpoloate from.
    pos_to : np.ndarray of float, shape(n_bad_sensors, 3)
        The positions to interpoloate.
    alpha : float
        Regularization parameter. Defaults to 1e-5.
    Returns
    -------
    interpolation : np.ndarray of float, shape(len(pos_from), len(pos_to))
        The interpolation matrix that maps good signals to the location
        of bad signals.
    References
    ----------
    [1] Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989).
        Spherical splines for scalp potential and current density mapping.
        Electroencephalography Clinical Neurophysiology, Feb; 72(2):184-7.
    """
    pos_from = pos_from.clone()
    pos_to = pos_to.clone()
    n_from = pos_from.shape[0]
    n_to = pos_to.shape[0]

    # normalize sensor positions to sphere
    pos_from = _torch_normalize_vectors(pos_from)
    pos_to = _torch_normalize_vectors(pos_to)

    # cosine angles between source positions
    cosang_from = torch.matmul(pos_from, pos_from.T)
    cosang_to_from = torch.matmul(pos_to, pos_from.T)
    G_from = _torch_calc_g(cosang_from)
    G_to_from = _torch_calc_g(cosang_to_from)
    assert G_from.shape == (n_from, n_from)
    assert G_to_from.shape == (n_to, n_from)

    if alpha is not None:
        G_from.flatten()[::len(G_from) + 1] += alpha

    device = G_from.device
    C = torch.vstack([
        torch.hstack([G_from, torch.ones((n_from, 1), device=device)]),
        torch.hstack([
            torch.ones((1, n_from), device=device),
            torch.as_tensor([[0]], device=device)])
    ])
    C_inv = torch.linalg.pinv(C)

    interpolation = torch.hstack([
        G_to_from,
        torch.ones((n_to, 1), device=device)
    ]).matmul(C_inv[:, :-1])
    assert interpolation.shape == (n_to, n_from)
    return interpolation


def _rotate_signals(X, rotations, sensors_positions_matrix, spherical=True):
    sensors_positions_matrix = sensors_positions_matrix.to(X.device)
    rot_sensors_matrices = [
        rotation.matmul(sensors_positions_matrix) for rotation in rotations
    ]
    if spherical:
        interpolation_matrix = torch.stack(
            [torch.as_tensor(
                _torch_make_interpolation_matrix(
                    sensors_positions_matrix.T, rot_sensors_matrix.T
                ), device=X.device
            ).float() for rot_sensors_matrix in rot_sensors_matrices]
        )
        return torch.matmul(interpolation_matrix, X)
    else:
        transformed_X = X.clone()
        sensors_positions = list(sensors_positions_matrix)
        for s, rot_sensors_matrix in enumerate(rot_sensors_matrices):
            rot_sensors_positions = list(rot_sensors_matrix.T)
            for time in range(X.shape[-1]):
                interpolator_t = Rbf(*sensors_positions, X[s, :, time])
                transformed_X[s, :, time] = torch.from_numpy(
                    interpolator_t(*rot_sensors_positions)
                )
        return transformed_X


def make_rotation_matrix(axis, angle, degrees=True):
    assert axis in ['x', 'y', 'z'], "axis should be either x, y or z."

    if isinstance(angle, (float, int, np.ndarray, list)):
        angle = torch.as_tensor(angle)

    if degrees:
        angle = angle * np.pi / 180

    device = angle.device
    zero = torch.zeros(1, device=device)
    rot = torch.stack([
        torch.as_tensor([1, 0, 0], device=device),
        torch.hstack([zero, torch.cos(angle), -torch.sin(angle)]),
        torch.hstack([zero, torch.sin(angle), torch.cos(angle)]),
    ])
    if axis == "x":
        return rot
    elif axis == "y":
        rot = rot[[2, 0, 1], :]
        return rot[:, [2, 0, 1]]
    else:
        rot = rot[[1, 2, 0], :]
        return rot[:, [1, 2, 0]]

def random_rotation(X, axis, max_degrees, sensors_positions_matrix,
                    spherical_splines, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    random_angles = torch.as_tensor(rng.uniform(
        low=0,
        high=1,
        size=X.shape[0]
    ), device=X.device) * 2 * max_degrees - max_degrees
    rots = [
        make_rotation_matrix(axis, random_angle, degrees=True)
        for random_angle in random_angles
    ]
    rotated_X = _rotate_signals(
        X, rots, sensors_positions_matrix, spherical_splines
    )
    return rotated_X

def get_standard_10_20_positions(raw_or_epoch=None, ordered_ch_names=None):
    """Returns standard 10-20 sensors position matrix (for instantiating
    RandomSensorsRotation for example).

    Parameters
    ----------
    raw_or_epoch : mne.io.Raw | mne.Epoch, optional
        Example of raw or epoch to retrive ordered channels list from. Need to
        be named as in 10-20. By default None.
    ordered_ch_names : list, optional
        List of strings representing the channels of the montage considered.
        The order has to be consistent with the order of channels in the input
        matrices that will be fed to `RandomSensorsRotation` transform. By
        default None.
    """
    assert raw_or_epoch is not None or ordered_ch_names is not None,\
        "At least one of raw_or_epoch and ordered_ch_names is needed."
    if ordered_ch_names is None:
        ordered_ch_names = raw_or_epoch.info['ch_names']
    ten_twenty_montage = make_standard_montage('standard_1020')
    positions_dict = ten_twenty_montage.get_positions()['ch_pos']
    positions_subdict = {
        k: positions_dict[k] for k in ordered_ch_names if k in positions_dict
    }
    return np.stack(list(positions_subdict.values())).T

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
def window_slice(x,rng, reduce_ratio=0.9): #ref (batch, time_steps, channel)
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = rng.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
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

def Window_Slicing_Circle(X, magnitude,window_size=1000, random_state=None, *args, **kwargs):
    rng = check_random_state(random_state)
    x = X.detach().cpu().numpy()
    dup_x = np.concatenate([x,x],axis=2)
    window_start = rng.randint(0, dup_x.shape[2] - window_size)
    new_x = x[:,:,window_start:window_start+window_size]
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
    
    random_warps = rng.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
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

def window_warp(x,rng, window_ratio=0.1, scales=[0.5, 2.]): #ref (batch, time_steps, channel)
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = rng.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = rng.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
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
    (Time_Warp, 0, 1),  # 3
    (Scaling, 0, 1),  # 4
    (Magnitude_Warp, 0, 1),  # 5
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

def get_augment(name):
    augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in TS_AUGMENT_LIST+ECG_AUGMENT_LIST+TS_ADD_LIST}
    return augment_dict[name]


def apply_augment(img, name, level, rd_seed=None):
    augment_fn, low, high = get_augment(name)
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
    def __init__(self, n, m, rd_seed=None):
        self.n = n
        self.m = m      # [0, 1]
        self.augment_list = TS_AUGMENT_LIST
        self.rng = check_random_state(rd_seed)
    def __call__(self, img):
        #print(img.shape)
        seq_len , channel = img.shape
        img = img.permute(1,0).view(1,channel,seq_len)
        ops = self.rng.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = float(self.m) * float(maxval - minval) + minval
            #print(val)
            img = op(img, val,random_state=self.rng)

        return img.permute(0,2,1).detach().view(seq_len,channel) #back to (len,channel)

class TransfromAugment:
    def __init__(self, names,m ,p=0.5,n=1, rd_seed=None):
        print(f'Using Fix transfroms {names}, m={m}, n={n}, p={p}')
        self.p = p
        self.m = m      # [0, 1]
        self.n = n
        self.names = names
        self.rng = check_random_state(rd_seed)
    def __call__(self, img):
        #print(img.shape)
        seq_len , channel = img.shape
        img = img.permute(1,0).view(1,channel,seq_len)
        select_names = self.rng.choices(self.names, k=self.n)
        for name in select_names:
            augment = get_augment(name)
            use_op = self.rng.random() < self.p
            if use_op:
                op, minval, maxval = augment
                val = float(self.m) * float(maxval - minval) + minval
                img = op(img, val,random_state=self.rng)
            else: #pass
                pass

        return img.permute(0,2,1).detach().view(seq_len,channel) #back to (len,channel)

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
    dataset = PTBXL(dataset_path='../../CWDA_research/CWDA/datasets/Datasets/ptbxl-dataset')
    print(dataset[0])
    print(dataset[0][0].shape)
    sample = dataset[50]
    x = sample[0]
    t = np.linspace(0, TimeS_dict['ptbxl'], 1000)
    label = sample[2]
    print(t.shape)
    print(x.shape)
    x_tensor = torch.from_numpy(x).float()
    plot_line(t,x)
    for name in TS_ADD_NAMES:
        print('='*10,name,'='*10)
        x_aug = apply_augment(x_tensor,name,0.5).numpy()
        print(x_aug.shape)
        plot_line(t,x_aug)
    randaug = RandAugment(1,0.5)
    name = 'random_time_mask'
    for i in range(3):
        #print('='*10,name,'='*10)
        x_aug = randaug(x_tensor,rd_seed=42).numpy()
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
