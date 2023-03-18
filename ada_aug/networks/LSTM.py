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
            out_pad, _out_len = rnn_utils.pad_packed_sequence(rnn_out, batch_first=True, total_length=x_shape[1])
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
    
#for wisdm
class HARModel(nn.Module): #Can not use with our current CAAP
    def __init__(self, n_hidden=128, n_layers=1, n_filters=64, n_channel=3,
                 n_classes=18, filter_size=5, drop_prob=0.5):
        super(HARModel, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size
        self.input_channels = n_channel
             
        self.conv1 = nn.Conv1d(self.input_channels, n_filters, filter_size)
        self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size)
        
        self.lstm1  = nn.LSTM(n_filters, n_hidden, n_layers)
        self.lstm2  = nn.LSTM(n_hidden, n_hidden, n_layers)
        
        self.fc = nn.Linear(n_hidden, n_classes)

        self.dropout = nn.Dropout(drop_prob)
        #init weight
        def init_weights(m):
            if type(m) == nn.LSTM:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            elif type(m) == nn.Conv1d or type(m) == nn.Linear:
                torch.nn.init.orthogonal_(m.weight)
                m.bias.data.fill_(0)
        self.apply(init_weights)
    
    def forward(self, x, seq_len=None):
        x_shape = x.shape
        x = self.extract_features(x,seq_len=seq_len)
        batch_size = x_shape[0]
        out = self.classify(x,batch_size)
        return out

    def extract_features(self, x, seq_len=None, pool=True):
        x_shape = x.shape
        if self.input_channels == x_shape[2]:
            x = x.transpose(1, 2) #(bs,len,ch) -> (bs, ch, len)
        x_shape = x.shape #(bs,ch,len)
        #x = x.view(batch_size, self.n_channel, -1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = x.view(8, -1, self.n_filters) 
        x, hidden = self.lstm1(x)
        x, hidden = self.lstm2(x,hidden)
        if pool:
            x = x.contiguous().view(-1, self.n_hidden)
        return x
    def pool_features(self, x):
        return x.contiguous().view(-1, self.n_hidden)
    def classify(self, features,batch_size): #may have bug
        features = self.dropout(features)
        features = self.fc(features)
        out = features.view(batch_size, -1, self.n_classes)[:,-1,:]
        return out
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        #if (train_on_gpu):
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        #else:
        #    hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
        #              weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        