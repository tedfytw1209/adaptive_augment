import torch
import torch.nn as nn

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)
Freq_dict = {
    'edfx' : 100,
    'ptbxl' : 100,
    'wisdm' : 20,
    'chapman' : 500,
}
TimeS_dict = {
    'edfx' : 30,
    'ptbxl' : 10,
    'wisdm' : 10,
    'chapman' : 10,
}

def soft_find(word,dic):
    for n in sorted(dic):
        if word in n:
            return dic[n]
    #raise error
    raise

class SleepStagerChambon2018(nn.Module):
    """Sleep staging architecture from [1]_.
    Convolutional neural network for sleep staging described in [1]_.
    Parameters: config
    ----------
    n_channels : int
        Number of EEG channels.
    sfreq : float
        EEG sampling frequency.
    n_conv_chs : int
        Number of convolutional channels. Set to 8 in [1]_.
    time_conv_size_s : float
        Size of filters in temporal convolution layers, in seconds. Set to 0.5
        in [1]_ (64 samples at sfreq=128).
    max_pool_size_s : float
        Max pooling size, in seconds. Set to 0.125 in [1]_ (16 samples at
        sfreq=128).
    n_classes : int
        Number of classes.
    input_size_s : float
        Size of the input, in seconds.
    dropout : float
        Dropout rate before the output dense layer.
    apply_batch_norm : bool
        If True, apply batch normalization after both temporal convolutional
        layers.
    References
    ----------
    .. [1] Chambon, S., Galtier, M. N., Arnal, P. J., Wainrib, G., &
           Gramfort, A. (2018). A deep learning architecture for temporal sleep
           stage classification using multivariate and multimodal time series.
           IEEE Transactions on Neural Systems and Rehabilitation Engineering,
           26(4), 758-769.
    """
    def __init__(self, config):
        super().__init__()
        n_channels = config.get('n_channels')
        data_set_name = config['dataset']
        sfreq = soft_find(data_set_name,Freq_dict)
        n_conv_chs = config.get('n_hidden',8)
        time_conv_size_s=0.5
        max_pool_size_s=0.125
        n_classes = config.get('n_output')
        input_size_s=soft_find(data_set_name,Freq_dict)
        dropout=config.get('fc_drop',0.25)
        apply_batch_norm=config.get('batch_norm',False)
        time_conv_size = int(time_conv_size_s * sfreq)
        max_pool_size = int(max_pool_size_s * sfreq)
        input_size = int(input_size_s * sfreq)
        pad_size = time_conv_size // 2
        self.n_channels = n_channels
        len_last_layer = self._len_last_layer(
            n_channels, input_size, max_pool_size, n_conv_chs)
        self.len_last_layer = len_last_layer
        print('CNN last layer len:', self.len_last_layer)
        self.fc.in_features = len_last_layer
        self.z_dim = len_last_layer
        if n_channels > 1:
            self.spatial_conv = nn.Conv2d(1, n_channels, (n_channels, 1))
        batch_norm = nn.BatchNorm2d if apply_batch_norm else nn.Identity
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1, n_conv_chs, (1, time_conv_size), padding=(0, pad_size)),
            batch_norm(n_conv_chs),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chs, n_conv_chs, (1, time_conv_size),
                padding=(0, pad_size)),
            batch_norm(n_conv_chs),
            nn.ReLU(),
            #nn.MaxPool2d((1, max_pool_size))
        )
        self.pool = nn.MaxPool2d(1, max_pool_size)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(len_last_layer, n_classes)
        )

    @staticmethod
    def _len_last_layer(n_channels, input_size, max_pool_size, n_conv_chs):
        return n_channels * (input_size // (max_pool_size ** 2)) * n_conv_chs

    def extract_features(self, x, seq_len=None, pool=True):
        x = x.transpose(1, 2) #(bs,len,ch) -> (bs, ch, len)
        x = x.unsqueeze(1) # (batch_size, 1, n_channels, n_times)
        if self.n_channels > 1:
            x = self.spatial_conv(x) # (batch_size, n_channels, 1, n_times)
            x = x.transpose(1, 2) # (batch_size, 1, n_channels, n_times)
        x = self.feature_extractor(x)
        if pool:
            x = self.pool(x)
            x = x.flatten(start_dim=1)
        return x
    
    def pool_features(self, x):
        return self.pool(x)
    
    def classify(self, features):
        return self.fc(features)

    def forward(self, x, seq_len=None):
        """Forward pass.
        Parameters
        ---------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        x = self.extract_features(x)
        return self.classify(x)
