import torch
import torch.nn as nn
import torch.nn.functional as F
import math

###############################################################################################

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

def create_pool(concat_pooling=True):
    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2), nn.Flatten()]
    return nn.Sequential(*layers)

def create_head1d(nf:int, nc:int, lin_ftrs=None, ps=0.5, bn_final:bool=False, bn:bool=True, act="relu", concat_pooling=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes; added bn and act here"
    lin_ftrs = [2*nf if concat_pooling else nf, nc] if lin_ftrs is None else [2*nf if concat_pooling else nf] + lin_ftrs + [nc] #was [nf, 256,nc]
    ps = list([ps])
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True) if act=="relu" else nn.ELU(inplace=True)] * (len(lin_ftrs)-2) + [nn.Identity()]
    #layers = [AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2), Flatten()]
    layers = []
    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
        layers += [
                nn.BatchNorm1d(ni),
                nn.Dropout(p),
                nn.Linear(ni, no),
                actn,]
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)

# Standard resnet

def conv(in_planes, out_planes, stride=1, kernel_size=3):
    "convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False)


class BasicBlock1d(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, kernel_size=[3,3], downsample=None):
        super().__init__()

        if(isinstance(kernel_size,int)): kernel_size = [kernel_size,kernel_size//2+1]

        self.conv1 = conv(inplanes, planes, stride=stride, kernel_size=kernel_size[0])
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes,kernel_size=kernel_size[1])
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck1d(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, kernel_size=3, downsample=None):
        super().__init__()
        
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size-1)//2, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet1d(nn.Module):
    '''1d adaptation of the torchvision resnet'''
    def __init__(self, block, layers, kernel_size=3, num_classes=2, input_channels=3, inplanes=64, fix_feature_dim=True, kernel_size_stem = None, stride_stem=2, pooling_stem=True, stride=2,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
        self.inplanes = inplanes
        super().__init__()
        layers_tmp = []
        self.input_channels = input_channels
        self.num_classes = num_classes
        if(kernel_size_stem is None):
            kernel_size_stem = kernel_size[0] if isinstance(kernel_size,list) else kernel_size
        #stem
        layers_tmp.append(nn.Conv1d(input_channels, inplanes, kernel_size=kernel_size_stem, stride=stride_stem, padding=(kernel_size_stem-1)//2,bias=False))
        layers_tmp.append(nn.BatchNorm1d(inplanes))
        layers_tmp.append(nn.ReLU(inplace=True))
        if(pooling_stem is True):
            layers_tmp.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        #backbone
        for i,l in enumerate(layers):
            if(i==0):
                layers_tmp.append(self._make_layer(block, inplanes, layers[0],kernel_size=kernel_size))
            else:
                layers_tmp.append(self._make_layer(block, inplanes if fix_feature_dim else (2**i)*inplanes, layers[i], stride=stride,kernel_size=kernel_size))
        
        self.pool = create_pool(concat_pooling=concat_pooling)
        nf = (inplanes if fix_feature_dim else (2**len(layers)*inplanes)) * block.expansion
        self.fc = create_head1d(nf, nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        self.fc.in_features = 2*nf if concat_pooling else nf
        self.z_dim = 2*nf if concat_pooling else nf
        self.feature_extractor = nn.Sequential( *layers_tmp)
        #head = create_head1d((inplanes if fix_feature_dim else (2**len(layers)*inplanes)) * block.expansion, nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        #layers_tmp.append(head)
        
        

    def _make_layer(self, block, planes, blocks, stride=1,kernel_size=3):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, kernel_size, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def get_layer_groups(self):
        return (self[6],self[-1])
    
    def get_output_layer(self):
        return self[-1][-1]
        
    def set_output_layer(self,x):
        self[-1][-1]=x
    def extract_features(self, x, seq_len=None, pool=True):
        #convert to 1dcnn ways (bs,len,ch) -> (bs,ch,len)
        #adaptive transpose
        x_shape = x.shape
        if self.input_channels == x_shape[2]:
            x = x.transpose(1, 2) #(bs,len,ch) -> (bs, ch, len)
        x = self.feature_extractor(x) #(bs,n_hidden,reduced len)
        if pool:
            x = self.pool(x)
        return x
    def pool_features(self, x):
        return self.pool(x)
    def classify(self, features):
        return self.fc(features)

    def forward(self, x, seq_len=None):
        x = self.extract_features(x)
        return self.classify(x)

def resnet1d18(kwargs):
    """Constructs a ResNet-18 model.
    """
    return ResNet1d(BasicBlock1d, [2, 2, 2, 2], **kwargs)

def resnet1d34(kwargs):
    """Constructs a ResNet-34 model.
    """
    return ResNet1d(BasicBlock1d, [3, 4, 6, 3], **kwargs)

def resnet1d50(kwargs):
    """Constructs a ResNet-50 model.
    """
    return ResNet1d(Bottleneck1d, [3, 4, 6, 3], **kwargs)

def resnet1d101(kwargs):
    """Constructs a ResNet-101 model.
    """
    return ResNet1d(Bottleneck1d, [3, 4, 23, 3], **kwargs)

def resnet1d152(kwargs):
    """Constructs a ResNet-152 model.
    """
    return ResNet1d(Bottleneck1d, [3, 8, 36, 3], **kwargs)


#original used kernel_size_stem = 8
def resnet1d_wang(kwargs):
    
    if(not("kernel_size" in kwargs.keys())):
        kwargs["kernel_size"]=[5,3]
    if(not("kernel_size_stem" in kwargs.keys())):
        kwargs["kernel_size_stem"]=7
    if(not("stride_stem" in kwargs.keys())):
        kwargs["stride_stem"]=1
    if(not("pooling_stem" in kwargs.keys())):
        kwargs["pooling_stem"]=False
    if(not("inplanes" in kwargs.keys())):
        kwargs["inplanes"]=128


    return ResNet1d(BasicBlock1d, [1, 1, 1], **kwargs)

def resnet1d(**kwargs):
    """Constructs a custom ResNet model.
    """
    return ResNet1d(BasicBlock1d, **kwargs)


###############################################################################################
# wide resnet adopted from fastai wrn

def noop(x): return x

def conv1d(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias=False) -> nn.Conv1d:
    "Create `nn.Conv1d` layer: `ni` inputs, `nf` outputs, `ks` kernel size. `padding` defaults to `k//2`."
    if padding is None: padding = ks//2
    return nn.Conv1d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias)

def _bn1d(ni, init_zero=False):
    "Batchnorm layer with 0 initialization"
    m = nn.BatchNorm1d(ni)
    m.weight.data.fill_(0 if init_zero else 1)
    m.bias.data.zero_()
    return m

def bn_relu_conv1d(ni, nf, ks, stride, init_zero=False):
    bn_initzero = _bn1d(ni, init_zero=init_zero)
    return nn.Sequential(bn_initzero, nn.ReLU(inplace=True), conv1d(ni, nf, ks, stride))

class BasicBlock1dwrn(nn.Module):
    def __init__(self, ni, nf, stride, drop_p=0.0, ks=3):
        super().__init__()
        if(isinstance(ks,int)):
            ks = [ks,ks//2+1]
        self.bn = nn.BatchNorm1d(ni)
        self.conv1 = conv1d(ni, nf, ks[0], stride)
        self.conv2 = bn_relu_conv1d(nf, nf, ks[0], 1)
        self.drop = nn.Dropout(drop_p, inplace=True) if drop_p else None
        self.shortcut = conv1d(ni, nf, ks[1], stride) if (ni != nf or stride>1) else noop #adapted to make it work for fix_feature_dim=True

    def forward(self, x):
        x2 = F.relu(self.bn(x), inplace=True)
        r = self.shortcut(x2)
        x = self.conv1(x2)
        if self.drop: x = self.drop(x)
        x = self.conv2(x) * 0.2
        return x.add_(r)

def _make_group(N, ni, nf, block, stride, drop_p,ks=3):
    return [block(ni if i == 0 else nf, nf, stride if i == 0 else 1, drop_p,ks=ks) for i in range(N)]

class WideResNet1d(nn.Sequential):
    def __init__(self, input_channels:int, num_groups:int, N:int, num_classes:int, k:int=1, drop_p:float=0.0, start_nf:int=16,fix_feature_dim=True,kernel_size=5,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
        super().__init__()
        n_channels = [start_nf]
        
        for i in range(num_groups): n_channels.append(start_nf if fix_feature_dim else start_nf*(2**i)*k)

        layers = [conv1d(input_channels, n_channels[0], 3, 1)]  # conv1 stem
        for i in range(num_groups):
            layers += _make_group(N, n_channels[i], n_channels[i+1], BasicBlock1dwrn, (1 if i==0 else 2), drop_p,ks=kernel_size)

        #layers += [nn.BatchNorm1d(n_channels[-1]), nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1),
        #           Flatten(), nn.Linear(n_channels[-1], num_classes)]
        head = create_head1d(n_channels[-1], nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        layers.append(head)
        
        super().__init__(*layers)
    
    def get_layer_groups(self):
        return (self[6],self[-1])
    
    def get_output_layer(self):
        return self[-1][-1]
    
    def set_output_layer(self,x):
        self[-1][-1] = x


def wrn1d_22(kwargs): return WideResNet1d(num_groups=3, N=3, k=6, drop_p=0.,**kwargs)
