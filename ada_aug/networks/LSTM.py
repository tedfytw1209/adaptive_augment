from pyexpat import features
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

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

class LSTM_ecg(nn.Module): #LSTM for time series 
    '''
    LSTM Module
    for self-supervised ECG: 2 layers and 512 hidden units
    concat-pooling layer, which concatenates the maximum of all LSTM outputs
    single hidden layer with 512 units including batch normalization and dropout
    '''
    def __init__(self, config):
        super().__init__()
        # params: "n_" means dimension
        self.n_layers = config['n_layers']   # number of layers
        self.config = config
        self.lstm = nn.LSTM(config['n_embed'], config['n_hidden'], num_layers=config['n_layers']
                , bidirectional=config['b_dir'], batch_first=True)
        self.dropout = nn.Dropout(config['fc_drop'])
        self.concat_pool = config.get('concat_pool',False)
        if self.concat_pool:
            self.concat_fc = nn.Sequential(*[nn.Dropout(config['fc_drop']),
                nn.Linear(3 * config['n_hidden'], config['n_hidden']),
                nn.BatchNorm1d(config['n_hidden']),
                nn.ReLU(),])
        self.fc = nn.Linear(config['n_hidden'], config['n_output'])

    def extract_features(self, texts, seq_lens):
        batch_size = texts.shape[0]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            texts, seq_lens.cpu(), batch_first=True)  # seq_len:128 [0]: lenght of each sentence
        rnn_out, (hidden, cell) = self.lstm(
            packed_embedded)  # bs X len X n_hidden
        out_pad, _out_len = rnn_utils.pad_packed_sequence(rnn_out, batch_first=True)
        if self.concat_pool:
            features = torch.cat([torch.max(out_pad,dim=1)[0],torch.mean(out_pad,dim=1),
                        out_pad[:,-1,:].view(batch_size,self.config['n_hidden'])],dim=-1)
            features = self.concat_fc(features)
        else:
            features = out_pad[:,-1,:].view(batch_size,self.config['n_hidden'])

        return features

    def classify(self, features):
        fc_out = self.fc(features)  # bs x d_out
        return fc_out

    def forward(self, x, seq_lens):
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

    def extract_features(self, x, seq_lens=None):
        x_shape = x.shape
        if self.input_channels == x_shape[1]:
            x = x.transpose(1, 2) #(bs,ch,len) -> (bs, len, ch)
        
        if seq_lens!=None:
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                x, seq_lens.cpu(), batch_first=True)  # seq_len:128 [0]: lenght of each sentence
        else:
            packed_embedded = x
        rnn_out, (hidden, cell) = self.lstm(
            packed_embedded)  # bs X len X n_hidden
        if seq_lens!=None:
            out_pad, _out_len = rnn_utils.pad_packed_sequence(rnn_out, batch_first=True)
        else:
            out_pad = rnn_out
        out_pad = out_pad.transpose(1, 2) # bs, n_hidden, len
        features = self.pool(out_pad) #bs, ch * (1+b_dir) * concat pool
        features = self.concat_fc(features) #bs, ch
        return features

    def classify(self, features):
        fc_out = self.fc(features)  # bs x d_out
        return fc_out

    def forward(self, x, seq_lens=None):
        x = self.extract_features(x, seq_lens)
        return self.classify(x)

class LSTM_modal(nn.Module): #LSTM for time series 
    '''
    LSTM Module
    for MODALS : n_hidden=128, b_dir=False, n_layers=1
    '''
    def __init__(self, config):
        super().__init__()
        # params: "n_" means dimension
        self.n_layers = config['n_layers']   # number of layers
        # number of hidden nodes
        self.rnn_hidden = config['n_hidden']//(
            2*config['n_layers'] if config['b_dir'] else config['n_layers'])

        self.rnn = self._cell(config['n_embed'], self.rnn_hidden,
                              config['n_layers'], config['rnn_drop'], config['b_dir'])
        
        if config.get('batch_norm',False):
            self.batch_norm = nn.BatchNorm1d(config['n_hidden'])
        else:
            self.batch_norm = nn.Identity()
        self.dropout = nn.Dropout(config['fc_drop'])
        self.fc = nn.Linear(config['n_hidden'], config['n_output'])

    def _cell(self,  n_embed, n_hidden, n_layers, drop_p, b_dir):
        cell = nn.LSTM(n_embed, n_hidden, n_layers, bidirectional=b_dir,batch_first=True)
        return cell

    def extract_features(self, texts, seq_lens):
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            texts, seq_lens.cpu(),batch_first=True)  # seq_len:128 [0]: lenght of each sentence
        rnn_out, (hidden, cell) = self.rnn(
            packed_embedded)  # 1 X bs X n_hidden
        features = hidden.permute(1, 0, 2).reshape(len(seq_lens), -1) #bs X n_hidden
        return features

    def classify(self, features):
        features = self.batch_norm(features)
        features = self.dropout(features)
        fc_out = self.fc(features)  # 1 x bs x d_out
#         softmax_out = F.softmax(fc_out, dim=-1)
        return fc_out

    def forward(self, x, seq_lens):
        x = self.extract_features(x, seq_lens)
        return self.classify(x)