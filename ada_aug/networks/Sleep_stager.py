import torch
import torch.nn as nn

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

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
        sfreq = config.get('sfreq')
        n_conv_chs=8
        time_conv_size_s=0.5
        max_pool_size_s=0.125
        n_classes = config.get('n_output')
        input_size_s=config.get('input_size_s',30)
        dropout=config.get('dropout',0.25)
        apply_batch_norm=config.get('batch_norm',False)
        time_conv_size = int(time_conv_size_s * sfreq)
        max_pool_size = int(max_pool_size_s * sfreq)
        input_size = int(input_size_s * sfreq)
        pad_size = time_conv_size // 2
        self.n_channels = n_channels
        len_last_layer = self._len_last_layer(
            n_channels, input_size, max_pool_size, n_conv_chs)
        self.len_last_layer = len_last_layer

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
            nn.MaxPool2d((1, max_pool_size))
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(len_last_layer, n_classes)
        )

    @staticmethod
    def _len_last_layer(n_channels, input_size, max_pool_size, n_conv_chs):
        return n_channels * (input_size // (max_pool_size ** 2)) * n_conv_chs

    def extract_features(self, x):
        x = x.unsqueeze(1) # (batch_size, 1, n_channels, n_times)
        if self.n_channels > 1:
            x = self.spatial_conv(x) # (batch_size, n_channels, 1, n_times)
            x = x.transpose(1, 2) # (batch_size, 1, n_channels, n_times)
        x = self.feature_extractor(x)
        x = x.flatten(start_dim=1)
        return x
    
    def classify(self, features):
        return self.fc(features)

    def forward(self, x):
        """Forward pass.
        Parameters
        ---------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        x = self.extract_features(x)
        return self.classify(x)
