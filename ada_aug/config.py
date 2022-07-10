OPS_NAMES = ['ShearX',
            'ShearY',
            'TranslateX',
            'TranslateY',
            'Rotate',
            'AutoContrast',
            'Invert',
            'Equalize',
            'Solarize',
            'Posterize',
            'Contrast',
            'Color',
            'Brightness',
            'Sharpness',
            'Cutout',
            'Flip',
            'Identity']

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

''' may add other transfroms
0 Identity
1 Jitter  [0.01, 0.5]
2 Time Warp knots,  {3, 4, 5}, [0.01, 0.5]
3 Window slice ratio [0.95, 0.6]
4 Window Warp Window ratio, window scales 0.1, [0.1, 2]
5 Scaling  [0.1,2.0]
6 Magnitude Warp knots,  {3, 4, 5}, [0.1, 2]
7 Permutation Max segments {3, 4, 5, 6}
8 Dropout p [0.05, 0.5]
'''

def get_warmup_config(dset):
    # multiplier, epoch
    config = {'svhn': (2, 2),
            'cifar10': (2, 5),
            'cifar100': (4, 5),
            'mnist': (1, 1),
            'imagenet': (2, 3)}
    if 'svhn' in dset:
        return config['svhn']
    elif 'cifar100' in dset:
        return config['cifar100']
    elif 'cifar10' in dset:
        return config['cifar10']
    elif 'mnist' in dset:
        return config['mnist']
    elif 'imagenet' in dset:
        return config['imagenet']
    else:
        return config['imagenet']


def get_search_divider(model_name):
    # batch size is too large if the search model is large
    # the divider split the update to multiple updates
    if model_name == 'wresnet40_2':
        return 32
    elif model_name == 'resnet50':
        return 128
    else:
        return 16
