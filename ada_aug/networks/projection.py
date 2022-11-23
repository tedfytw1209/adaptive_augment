import torch
import torch.nn as nn
from config import OPS_NAMES,TS_OPS_NAMES
from operation_tseries import GOOD_ECG_NAMES, TS_ADD_NAMES,ECG_OPS_NAMES,ECG_NOISE_NAMES

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
            select = binomial.sample([X.shape[0]]).long()
            print(self.masks[select].shape)
            out = X * self.masks[select].to(X.device) * self.ratios[select].to(X.device) # (bs,in_dim) * (bs,in_dim) * (bs,1)
            return out

        return X

class AlphaAdd(nn.Module):
    def __init__(self, fea_len: int = 128, label_len: int = 32):
        super(AlphaAdd, self).__init__()
        assert fea_len==label_len
        self.fea_len = fea_len
        self.label_len = label_len

    def forward(self, X):
        bs, n_hidden = X.shape
        #print('Agg x shape: ',X.shape)
        alpha = torch.ones(bs,1).cuda() * 0.5
        #print(alpha) #!tmp
        if self.training:
            '''binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            return X * binomial.sample(X.size()) * (1.0/(1-self.p))'''
            #out = X * self.masks[select].to(X.device) * self.ratios[select].to(X.device) # (bs,in_dim) * (bs,in_dim) * (bs,1)
            fea_x, label_x = torch.split(X, [self.fea_len, self.label_len], dim=1)
            out = fea_x * alpha + label_x * (1.0-alpha) #embedding mix
            return out
        else:
            fea_x, label_x = torch.split(X, [self.fea_len, self.label_len], dim=1)
            out = fea_x * alpha + label_x * (1.0-alpha) #embedding mix
            return out

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
    def __init__(self, in_features, n_layers, n_hidden=128,label_num=0,label_embed=0, augselect='', proj_addition=0, 
        feature_mask="", input_act=False,proj_b=True,embed_b=True):
        super(Projection_TSeries, self).__init__()
        self.ops_names = TS_OPS_NAMES.copy()
        if augselect.startswith('goodtrans'): #only use good transfrom
            self.ops_names = GOOD_ECG_NAMES.copy()
        if 'tsadd' in augselect:
            self.ops_names = self.ops_names + TS_ADD_NAMES.copy()
        if 'ecg_noise' in augselect:
            self.ops_names = ECG_NOISE_NAMES.copy()
        elif 'ecg' in augselect:
            self.ops_names = self.ops_names + ECG_OPS_NAMES.copy()
        self.ops_len = len(self.ops_names)
        print('Projection Using ',self.ops_names)
        print('In_features: ',in_features) #already add y_feature_len if needed
        proj_out = 2*self.ops_len + proj_addition
        self.label_embed = None
        self.feature_embed = None
        if label_num>0 and label_embed>0:
            #self.label_embed = nn.Sequential(nn.Linear(label_num, label_embed),nn.ReLU()) #old ver
            self.label_embed = nn.Linear(label_num, label_embed, bias=embed_b) #10/10 change
            n_label = label_embed
        elif label_num>0:
            n_label = label_num
        else:
            n_label = 0
        self.class_adapt = int(n_label>0)
        self.n_layers = n_layers
        layers = []
        self.input_act = input_act
        self.feature_mask = feature_mask
        self.proj_b = proj_b
        self.embed_b = embed_b
        #
        if self.n_layers > 0:
            layers = [] 
            if feature_mask=='dropout':
                self.feature_embed = nn.Sequential(nn.Linear(in_features-n_label, n_hidden,bias=embed_b), nn.ReLU())
                layers += [nn.Dropout(p=0.5)]
            elif feature_mask=='select':
                self.feature_embed = nn.Sequential(nn.Linear(in_features-n_label, n_hidden,bias=embed_b), nn.ReLU())
                layers += [SelectDropout(p=0.5,fea_len=n_hidden,label_len=n_label)] #custom dropout
            elif feature_mask=='average':
                self.feature_embed = nn.Sequential(nn.Linear(in_features-n_label, n_hidden,bias=embed_b))
                layers += [AlphaAdd(fea_len=n_hidden,label_len=n_label)]
                n_label = 0
            for _ in range(self.n_layers-1):
                layers.append(nn.Linear(n_hidden + n_label, n_hidden,bias=proj_b))
                layers.append(nn.ReLU())
                n_label = 0
            layers.append(nn.Linear(n_hidden + n_label, proj_out,bias=proj_b))
        else:
            if feature_mask=='dropout':
                layers += [nn.Dropout(p=0.5)]
            elif feature_mask=='select':
                layers += [SelectDropout(p=0.5,fea_len=in_features-label_embed,label_len=n_label)]
            elif feature_mask == 'classonly':
                in_features = label_embed
            layers += [nn.Linear(in_features, proj_out,bias=proj_b)]
        self.projection = nn.Sequential(*layers)

    def forward(self, x,y=None):
        if self.feature_embed!=None:
            x = self.feature_embed(x)
        
        if not self.class_adapt:
            agg_x = x
        elif self.feature_mask=='classonly':
            y_tmp = self.label_embed(y)
            agg_x = y_tmp
            #print('class only y: ',agg_x)
        elif self.label_embed!=None:
            y_tmp = self.label_embed(y)
            agg_x = torch.cat([x,y_tmp], dim=1) #feature dim
            #print(x.shape, y_tmp.shape)
        else:
            agg_x = torch.cat([x,y], dim=1) #feature dim
        
        if self.input_act:
            agg_x = nn.functional.relu(agg_x)
        return self.projection(agg_x)