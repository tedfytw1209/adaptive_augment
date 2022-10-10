import torch
import torch.nn as nn
from config import OPS_NAMES,TS_OPS_NAMES
from operation_tseries import TS_ADD_NAMES,ECG_OPS_NAMES

class SelectDropout(nn.Module):
    def __init__(self, p: float = 0.5, fea_len: int = 128, label_len: int = 32):
        super(SelectDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.masks = torch.zeros((2, fea_len+label_len)).float()
        self.masks[0,:fea_len] = 1.0
        self.masks[1,fea_len:] = 1.0
        self.ratios = torch.zeros((2,1)).float()
        self.ratios[0,0] = (fea_len + label_len) / fea_len
        self.ratios[1,0] = (fea_len + label_len) / label_len

    def forward(self, X):
        if self.training:
            '''binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            return X * binomial.sample(X.size()) * (1.0/(1-self.p))'''
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            select = binomial.sample(X.shape[0])
            out = X * self.masks[select] * self.ratios[select] # (bs,in_dim) * (bs,in_dim) * (bs,1)
            return out

        return X

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
    def __init__(self, in_features, n_layers, n_hidden=128,label_num=0,label_embed=0, augselect='', proj_addition=0, feature_mask=""):
        super(Projection_TSeries, self).__init__()
        self.ops_names = TS_OPS_NAMES.copy()
        if 'tsadd' in augselect:
            self.ops_names += TS_ADD_NAMES.copy()
        if 'ecg' in augselect:
            self.ops_names += ECG_OPS_NAMES.copy()
        self.ops_len = len(self.ops_names)
        print('Projection Using ',self.ops_names)
        print('In_features: ',in_features) #already add y_feature_len if needed
        proj_out = 2*self.ops_len + proj_addition
        self.label_embed = None
        if label_num>0 and label_embed>0:
            self.label_embed = nn.Sequential(nn.Linear(label_num, label_embed), nn.ReLU()) #10/10 change
        self.n_layers = n_layers
        layers = []
        if feature_mask=='dropout':
            layers += [nn.Dropout(p=0.5)]
        elif feature_mask=='select':
            layers += [SelectDropout(p=0.5)] #custom dropout
        if self.n_layers > 0:
            layers += [nn.Linear(in_features, n_hidden), nn.ReLU()]
            for _ in range(self.n_layers-1):
                layers.append(nn.Linear(n_hidden, n_hidden))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_hidden, proj_out))
        else:
            layers += [nn.Linear(in_features, proj_out)]
        self.projection = nn.Sequential(*layers)

    def forward(self, x,y=None):
        if not torch.is_tensor(y):
            agg_x = x
        elif self.label_embed!=None:
            y_tmp = self.label_embed(y)
            agg_x = torch.cat([x,y_tmp], dim=1) #feature dim
        else:
            agg_x = torch.cat([x,y], dim=1) #feature dim
        return self.projection(agg_x)