from __future__ import print_function
# from __future__ import absolute_import
import torch

from torch import nn
from torch.nn import DataParallel

from .resnet import ResNet
from .wideresnet import WideResNet
from .LSTM import LSTM_ecg,LSTM_modal,LSTM_ptb
from .LSTM_attention import LSTM_attention
from .Sleep_stager import SleepStagerChambon2018
from .resnet1d import resnet1d_wang,resnet1d101
from .xresnet1d import xresnet1d101
from .MF_transformer import MF_Transformer

def count_parameters(model):
    temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f' |Trainable parameters: {temp}')

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

def get_model_tseries(model_name='lstm', num_class=10, n_channel=3, use_cuda=True, data_parallel=False, dataset='', max_len=5000,hz=100,
    addition_config={}):
    name = model_name
    config = {'n_output':num_class,'n_embed':n_channel,'rnn_drop': 0.2,'fc_drop': 0.5,'max_len':max_len,'hz':hz}
    config.update(addition_config)
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
    elif model_name == 'lstm_ptb':
        n_hidden = 128
        model_config = {
                  'n_hidden': n_hidden,
                  'n_layers': 2,
                  'b_dir': False,
                  'concat_pool': True,
                  'rnn_drop': 0.25,
                  'fc_drop': 0.5}
        net = LSTM_ptb
    elif model_name == 'mf_trans':
        n_hidden = 256 # old mf_trans before 1/17=128, new mf_trans=256
        model_config = {
                  'n_hidden': n_hidden,
                  'n_layers': 5,
                  'n_head': 8, #tmp params
                  'n_dff': n_hidden*2, #tmp params
                  'b_dir': False,
                  'concat_pool': True,
                  'rnn_drop': 0.1,
                  'fc_drop': 0.5}
        net = MF_Transformer
    elif model_name == 'mf_trans2':
        n_hidden = 512
        model_config = {
                  'n_hidden': n_hidden,
                  'n_layers': 5,
                  'n_head': 8, #tmp params
                  'n_dff': n_hidden*2, #tmp params
                  'b_dir': False,
                  'concat_pool': True,
                  'rnn_drop': 0.1,
                  'fc_drop': 0.5}
        net = MF_Transformer
    elif model_name == 'lstm_atten':
        n_hidden = 512
        model_config = {
                  'n_hidden': n_hidden,
                  'n_layers': 1,
                  'b_dir': True,
                  'rnn_drop': 0.2,
                  'fc_drop': 0.5}
        net = LSTM_attention
    elif model_name == 'resnet_wang':
        n_hidden = 128
        config = {
                  'input_channels': n_channel,
                  'inplanes': n_hidden,
                  'num_classes': num_class,
                  'kernel_size': 5,
                  'lin_ftrs_head': [n_hidden], #8/17 add
                  'ps_head': 0.5}
        model_config = {}
        net = resnet1d_wang
    elif model_name == 'resnet101':
        n_hidden = 128
        config = {
                  'input_channels': n_channel,
                  'inplanes': n_hidden,
                  'num_classes': num_class,
                  'kernel_size': 5,
                  'lin_ftrs_head': [n_hidden], #8/17 add
                  'ps_head': 0.5}
        model_config = {}
        net = resnet1d101
    elif model_name == 'xresnet101':
        #conf_fastai_xresnet1d101 = {'modelname':'fastai_xresnet1d101', 'modeltype':'fastai_model', 'parameters':dict()}
        #elif(self.name.startswith("fastai_xresnet1d101")):
        #    model = xresnet1d101(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        n_hidden = 128
        config = {
                  'input_channels': n_channel,
                  'inplanes': n_hidden,
                  'num_classes': num_class,
                  'kernel_size': 5,
                  'lin_ftrs_head': [n_hidden], #8/17 add
                  'ps_head': 0.5}
        model_config = {}
        net = xresnet1d101
    elif model_name == 'inception':
        n_hidden = 128
        config = {
                  'input_channels': n_channel,
                  'inplanes': n_hidden,
                  'num_classes': num_class,
                  'kernel_size': 5,
                  'lin_ftrs_head': [n_hidden], #8/17 add
                  'ps_head': 0.5}
        model_config = {}
        net = resnet1d_wang
    elif model_name == 'fcn_wang':
        n_hidden = 128
        config = {
                  'input_channels': n_channel,
                  'inplanes': n_hidden,
                  'num_classes': num_class,
                  'kernel_size': 5,
                  'lin_ftrs_head': [n_hidden], #8/17 add
                  'ps_head': 0.5}
        model_config = {}
        net = resnet1d_wang
    elif model_name == 'cnn_sleep': #with problems!!!
        model_config = {
                  'dataset': dataset,
                  'batch_norm': True,
                  'fc_drop': 0.25,
                  }
        net = SleepStagerChambon2018
    else:
        raise NameError('no model named, %s' % name)
    config.update(model_config)
    model = net(config)
    if data_parallel:
        model = model.cuda()
        model = DataParallel(model)
    else:
        if use_cuda:
            model = model.cuda()
    print('\n### Model ###')
    print(f'=> {model_name}')
    count_parameters(model)
    print(f'embedding=> {n_hidden}')
    print(model)
    return model