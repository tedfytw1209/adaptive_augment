from __future__ import print_function
# from __future__ import absolute_import
import torch

from torch import nn
from torch.nn import DataParallel

from .resnet import ResNet
from .wideresnet import WideResNet
from .LSTM import LSTM_ecg,LSTM_modal
from .LSTM_attention import LSTM_attention
from .Sleep_stager import SleepStagerChambon2018

def get_model(model_name='wresnet40_2', num_class=10, n_channel=3, use_cuda=True, data_parallel=False):
    name = model_name

    if name == 'resnet50':
        model = ResNet(dataset='imagenet', n_channel=n_channel, depth=50, num_classes=num_class, bottleneck=True)
    elif name == 'resnet200':
        model = ResNet(dataset='imagenet', n_channel=n_channel, depth=200, num_classes=num_class, bottleneck=True)
    elif name == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=0.0, num_classes=num_class)
    elif name == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=0.0, num_classes=num_class)
    else:
        raise NameError('no model named, %s' % name)

    if data_parallel:
        model = model.cuda()
        model = DataParallel(model)
    else:
        if use_cuda:
            model = model.cuda()
    return model

def get_model_tseries(model_name='lstm', num_class=10, n_channel=3, use_cuda=True, data_parallel=False):
    name = model_name
    config = {'n_output':num_class,'n_embed':n_channel,'rnn_drop': 0.2,'fc_drop': 0.5}
    if model_name == 'lstm':
        n_hidden = 128
        model_config = {'n_hidden': n_hidden,
                  'n_layers': 1,
                  'b_dir': False,
                  }
        net = LSTM_modal
    elif model_name == 'lstm_ecg':
        n_hidden = 512
        model_config = {
                  'n_hidden': n_hidden,
                  'n_layers': 2,
                  'b_dir': False,
                  }
        net = LSTM_ecg
    elif model_name == 'lstm_atten':
        n_hidden = 512
        model_config = {
                  'n_hidden': n_hidden,
                  'n_layers': 1,
                  'b_dir': True,
                    }
        net = LSTM_attention
    elif model_name == 'cnn_sleep': #with problems!!!
        model_config = {
                  'sfreq': 100,
                  'batch_norm': True,
                  'dropout': 0.25,
                  }
        net = SleepStagerChambon2018
    else:
        raise NameError('no model named, %s' % name)
    config.update(model_config)
    model = net(config=config)
    if data_parallel:
        model = model.cuda()
        model = DataParallel(model)
    else:
        if use_cuda:
            model = model.cuda()
    print('\n### Model ###')
    print(f'=> {model_name}')
    print(f'embedding=> {n_hidden}')
    return model