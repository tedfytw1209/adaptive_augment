import torch
import torch.nn as nn
from config import OPS_NAMES,TS_OPS_NAMES
from operation_tseries import TS_ADD_NAMES,ECG_OPS_NAMES

class Projection(nn.Module):
    def __init__(self, in_features, n_layers, n_hidden=128):
        super(Projection, self).__init__()
        self.n_layers = n_layers
        if self.n_layers > 0:
            layers = [nn.Linear(in_features, n_hidden), nn.ReLU()]
            for _ in range(self.n_layers-1):
                layers.append(nn.Linear(n_hidden, n_hidden))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_hidden, 2*len(OPS_NAMES)))
        else:
            layers = [nn.Linear(in_features, 2*len(OPS_NAMES))]
        self.projection = nn.Sequential(*layers)

    def forward(self, x,y=None):
        if torch.is_tensor(y):
            agg_x = torch.cat([x,y], dim=1) #feature dim
        else:
            agg_x = x
        return self.projection(agg_x)

class Projection_TSeries(nn.Module):
    def __init__(self, in_features, n_layers, n_hidden=128,label_num=None,label_embed=None, augselect=''):
        super(Projection_TSeries, self).__init__()
        self.ops_names = TS_OPS_NAMES.copy()
        if 'tsadd' in augselect:
            self.ops_names += TS_ADD_NAMES.copy()
        if 'ecg' in augselect:
            self.ops_names += ECG_OPS_NAMES.copy()
        self.ops_len = len(self.ops_names)
        print('Projection Using ',self.ops_names)
        print('In_features: ',in_features)
        if label_embed!=None:
            self.label_embed = nn.Linear(label_num, label_embed)
        self.n_layers = n_layers
        if self.n_layers > 0:
            layers = [nn.Linear(in_features, n_hidden), nn.ReLU()]
            for _ in range(self.n_layers-1):
                layers.append(nn.Linear(n_hidden, n_hidden))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_hidden, 2*self.ops_len))
        else:
            layers = [nn.Linear(in_features, 2*self.ops_len)]
        self.projection = nn.Sequential(*layers)

    def forward(self, x,y=None):
        if torch.is_tensor(y):
            agg_x = torch.cat([x,y], dim=1) #feature dim
        else:
            agg_x = x
        return self.projection(agg_x)