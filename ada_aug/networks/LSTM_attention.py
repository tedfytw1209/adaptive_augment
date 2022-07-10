import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class LSTM_attention(nn.Module):

    def __init__(self,config):
        super(LSTM_attention, self).__init__()
        self.n_layers = config['n_layers']   # number of layers
        self.config = config
        self.hidden_dim = config['n_hidden']
        self.embedding_dim = config['n_embed']
        self.num_class = config['n_output']
        self.bidirect = config['b_dir']
        self.dropout = config['fc_drop']

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.n_layers, bidirectional=self.bidirect, batch_first=True)

        # The linear layer that maps from hidden dim space to attention weight
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, 1, bias=False),
        )

        # Single head output for num classes
        self.fc = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.num_class)
            )

    def attention_net_with_w(self, lstm_out):
        """
        Args:
            lstm_out: [batch_size, time_step, hidden_dim * 2]
        Returns:
            h_new:    [batch_size, hidden_dim]
            alpha:    [batch_size, 1, time_step]
        """
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)

        # h: batch_size, time_step, hidden_dim
        h = lstm_tmp_out[0] + lstm_tmp_out[1]

        # m: batch_size, time_step, hidden_dim
        m = torch.tanh(h)

        # alpha: batch_size, 1, time_step
        alpha = F.softmax(self.attention_layer(m).transpose(1, 2), dim=-1)

        # r: batch_size, 1, hidden_dim
        r = torch.bmm(alpha, h)

        # h_new: batch_size, hidden_dim
        h_new = torch.tanh(r.squeeze(1))

        return h_new, alpha
    
    def extract_features(self, sequence, sequence_len):
        sequence_pack = rnn_utils.pack_padded_sequence(sequence, sequence_len.cpu(), batch_first=True)
        lstm_out, (h, c) = self.lstm(sequence_pack)
        out_pad, _out_len = rnn_utils.pad_packed_sequence(lstm_out, batch_first=True)
        atten_out, alpha = self.attention_net_with_w(out_pad)
        return atten_out

    def classify(self, features):
        fc_out = self.fc(features)  # bs x d_out
        return fc_out

    def forward(self, x, seq_lens):
        x = self.extract_features(x, seq_lens)
        return self.classify(x)


