from pyexpat import features
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import math
import copy
from ecgdetectors import Detectors
import numpy as np

class AdaptiveConcatPoolRNN(nn.Module):
    def __init__(self, bidirectional):
        super().__init__()
        self.bidirectional = bidirectional
    def forward(self,x):
        #input shape bs, ch, ts
        t1 = nn.AdaptiveAvgPool1d(1)(x)
        t2 = nn.AdaptiveMaxPool1d(1)(x)
        
        if(self.bidirectional is False):
            t3 = x[:,:,-1]
        else:
            channels = x.size()[1]
            t3 = torch.cat([x[:,:channels,-1],x[:,channels:,0]],1)
        out=torch.cat([t1.squeeze(-1),t2.squeeze(-1),t3],1) #output shape bs, 3*ch
        return out

class LastPoolRNN(nn.Module):
    def __init__(self, bidirectional):
        super().__init__()
        self.bidirectional = bidirectional
    def forward(self,x):
        #input shape bs, ch, ts
        
        if(self.bidirectional is False):
            t3 = x[:,:,-1]
        else:
            channels = x.size()[1]
            t3 = torch.cat([x[:,:channels,-1],x[:,channels:,0]],1)
        out=t3 #output shape bs, ch * b_dir
        return out

#position encode
class PositionalEncoding(nn.Module):
    #assume x=(bs,seq_len,ch)
    def __init__(self, d_model, dropout, max_len=5000):
        print('Position Encode max len: ',max_len)
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) #(max_len,d_model)
        position = torch.arange(0,max_len).unsqueeze(1) #(max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
            -(math.log(10000.0) / d_model)) #(d_model/2)
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0) #(1,max_len,d_model)
        self.register_buffer('pe',pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)
#multihead atten
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class Encoder(nn.Module):
    def __init__(self, layer, N) -> None:
        super(Encoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask=None):
        for (i,layer) in enumerate(self.layers):
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    def forward(self, x, mask=None):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

#atten
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
#multi-head atten
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

#finial transformer
class MF_Transformer(nn.Module): #LSTM for time series 
    '''
    Multi-Feature transformer
    backbone from GeoECG
    N = 5 same layers, each layer contains two sub-layers: a multi-head self-attention model and a fully connected
    feed-forward network, Residual connection and normalization are added in each sub-layer.
    1D convolutional and softmax layers for the output. ???
    '''
    def __init__(self, config):
        super().__init__()
        self.input_channels = config['n_embed']
        self.z_dim = config['n_hidden']
        self.num_classes = config['n_output']
        self.n_layers = config['n_layers']   # number of layers
        self.bidir_factor = 1 + int(config['b_dir'])
        self.config = config
        c = copy.deepcopy
        n_input = config['n_embed'] * config['hz'] #not good impl
        d_model = config['n_hidden']
        n_embed = config['n_hidden']
        N = config['n_layers']
        h = config['n_head']
        drop_p = config['rnn_drop']
        d_ff = config['n_dff']
        ori_max_len = config['max_len']
        attn = MultiHeadedAttention(h,d_model)
        ff = PositionwiseFeedForward(d_model,d_ff,drop_p)
        #model: input_embed, pos_embed, atten_model, fc
        seg_config = config['seg_config']
        self.segmentation = Segmentation(**seg_config,origin_max_len=ori_max_len)
        max_len = self.segmentation.max_len
        self.input_embed = nn.Linear(n_input, n_embed,bias=False) #linear as embed
        self.position = PositionalEncoding(d_model,drop_p,max_len=max_len)
        self.atten_encoder = Encoder(EncoderLayer(d_model,c(attn),c(ff),drop_p),N)
        #pool or not
        self.concat_pool = config.get('concat_pool',False)
        if self.concat_pool:
            mult_factor = 3
            self.pool = AdaptiveConcatPoolRNN(False)
        else:
            mult_factor = 1
            self.pool = LastPoolRNN(False)
        self.concat_fc = nn.Sequential(*[
                nn.BatchNorm1d(mult_factor * d_model),
                nn.Dropout(drop_p),
                nn.Linear(mult_factor * d_model, d_model),
                nn.ReLU(),])
        self.fc = nn.Sequential(*[
                nn.BatchNorm1d(d_model),
                nn.Dropout(config['fc_drop']),
                nn.Linear(d_model, config['n_output'])])
        self.fc.in_features = d_model

    def extract_features(self, x, seq_lens=None, pool=True):
        x_shape = x.shape
        if self.input_channels == x_shape[1]:
            x = x.transpose(1, 2) #(bs,ch,len) -> (bs, len, ch)
        bs, slen, ch = x.shape #x=(bs,len,n_hidden)
        if seq_lens==None:
            seq_lens = torch.full((bs,),slen).long()
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(
        #    x, seq_lens.cpu(), batch_first=True)  # seq_len:128 [0]: lenght of each sentence
        seg_x,seg_len = self.segmentation(x,seq_lens)
        x_embed = self.input_embed(seg_x)
        x_posemd = self.position(x_embed)
        x_encoded = self.atten_encoder(x_posemd)
        #rnn_out, (hidden, cell) = self.lstm(
        #    packed_embedded)  # bs X len X n_hidden
        #out_pad, _out_len = rnn_utils.pad_packed_sequence(rnn_out, batch_first=True)
        features = x_encoded.transpose(1, 2) #change to input shape bs, ch, ts
        if pool:
            features = self.pool_features(features) #bs, ch * (1+b_dir) * concat pool
        return features

    def pool_features(self, features):
        features = self.pool(features) #bs, ch * (1+b_dir) * concat pool
        features = self.concat_fc(features) #bs, ch
        return features
    
    def classify(self, features):
        fc_out = self.fc(features)  # bs x d_out
        return fc_out

    def forward(self, x, seq_lens=None):
        x = self.extract_features(x, seq_lens)
        return self.classify(x)

class LSTM_ptb(nn.Module): #LSTM for PTBXL
    '''
    LSTM Module
    for self-supervised ECG: 2 layers and 256 hidden units
    concat-pooling layer, which concatenates the max/mean of all LSTM outputs
    single hidden layer with 128 units including batch normalization and dropout
    '''
    def __init__(self, config):
        super().__init__()
        # params: "n_" means dimension
        self.input_channels = config['n_embed']
        self.z_dim = config['n_hidden']
        self.num_classes = config['n_output']
        self.n_layers = config['n_layers']   # number of layers
        self.bidir_factor = 1 + int(config['b_dir'])
        self.config = config
        self.lstm = nn.LSTM(config['n_embed'], 2*config['n_hidden'], num_layers=config['n_layers']
                , bidirectional=config['b_dir'], batch_first=True)
        self.concat_pool = config.get('concat_pool',False)
        if self.concat_pool:
            mult_factor = 3
            self.pool = AdaptiveConcatPoolRNN(config['b_dir'])
        else:
            mult_factor = 1
            self.pool = LastPoolRNN(config['b_dir'])
        self.concat_fc = nn.Sequential(*[
                nn.BatchNorm1d(self.bidir_factor * mult_factor * 2*config['n_hidden']),
                nn.Dropout(config['rnn_drop']),
                nn.Linear(self.bidir_factor * mult_factor * 2*config['n_hidden'], config['n_hidden']),
                nn.ReLU(),])
        self.fc = nn.Sequential(*[
                nn.BatchNorm1d(config['n_hidden']),
                nn.Dropout(config['fc_drop']),
                nn.Linear(config['n_hidden'], config['n_output'])])
        self.fc.in_features = config['n_hidden']

    def extract_features(self, x, seq_lens=None, pool=True):
        x_shape = x.shape
        if self.input_channels == x_shape[1]:
            x = x.transpose(1, 2) #(bs,ch,len) -> (bs, len, ch)
        
        if seq_lens!=None:
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                x, seq_lens.cpu(), batch_first=True,enforce_sorted=False)  # seq_len:128 [0]: lenght of each sentence
        else:
            packed_embedded = x
        rnn_out, (hidden, cell) = self.lstm(
            packed_embedded)  # bs X len X n_hidden
        if seq_lens!=None:
            out_pad, _out_len = rnn_utils.pad_packed_sequence(rnn_out, batch_first=True)
        else:
            out_pad = rnn_out
        features = out_pad.transpose(1, 2) # bs, n_hidden, len
        if pool:
            features = self.pool(features) #bs, ch * (1+b_dir) * concat pool
            features = self.concat_fc(features) #bs, ch
        return features
    def pool_features(self, features):
        features = self.pool(features) #bs, ch * (1+b_dir) * concat pool
        features = self.concat_fc(features) #bs, ch
        return features
    
    def classify(self, features):
        fc_out = self.fc(features)  # bs x d_out
        return fc_out

    def forward(self, x, seq_lens=None):
        x = self.extract_features(x, seq_lens)
        return self.classify(x)

class Segmentation(nn.Module): #segment data for Transfromer
    def __init__(self, seg_ways='fix', rr_method='pan',pw_len=0.4,tw_len=0.6,hz=100,origin_max_len=5000):
        #rr_method: fix, rpeak
        super().__init__()
        self.seg_ways = seg_ways
        self.rr_method = rr_method
        self.pw_len = pw_len * hz
        self.tw_len = tw_len * hz
        self.hz = hz
        self.detect_lead = 1 #normal use lead II
        if self.seg_ways=='rpeak':
            self.detectors = Detectors(self.hz) #need input ecg: (seq_len)
            #detector
            if rr_method=='pan':
                self.detect_func = self.detectors.pan_tompkins_detector
        elif self.seg_ways=='fix':
            self.detect_func = None
        self.origin_max_len = origin_max_len
        self.max_len = int(self.origin_max_len / self.hz)
    
    def forward(self,x, seq_lens=None):
        bs, slen, ch = x.shape
        new_len = int(slen / self.hz) #max len after transform
        new_ch = ch * self.hz
        if seq_lens==None:
            seq_lens = torch.full((bs),slen).long()
        
        if self.detect_func==None:
            tmp_x = x.reshape(bs,new_len,new_ch)
            new_seq_lens = (seq_lens / self.hz).long() #real len after transform
        else:
            print('x shape: ',x.shape) #!tmp
            x_single = x[:,:,self.detect_lead].detach().cpu().numpy()
            new_seq_lens = torch.zeros(bs)
            tmp_x = []
            for i,(x_each,slen_each) in enumerate(zip(x_single,seq_lens)): #each x = (seq_len)
                rpeaks_array = self.detect_func(x_each[:slen_each])
                new_seq_lens[i] = len(rpeaks_array)
                new_x = torch.zeros(new_len,new_ch)
                for p,peak in enumerate(rpeaks_array):
                    x1 = np.clip(peak - self.pw_len , 0, slen)
                    x2 = np.clip(peak + self.tw_len , 0, slen)
                    new_x[p] = x_each[x1:x2,:].reshape(-1)
                tmp_x.append(new_x)
            tmp_x = torch.stack(tmp_x, dim=0).to(x.device)
            print('segmented shape: ',tmp_x.shape) #!tmp
        
        return tmp_x, new_seq_lens
