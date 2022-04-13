import torch
from torch import nn

from models.wav2vec.wav2vec import Wav2Vec2Model

"""
Wav2Keyword model
cf https://github.com/qute012/Wav2Keyword
"""


class Wav2Keyword(nn.Module):
    def __init__(self, n_class=12, encoder_hidden_dim=768, cfg=None, state_dict=None):
        super(Wav2Keyword, self).__init__()
        self.n_class = n_class

        self.w2v_encoder = Wav2Vec2Model(cfg)
        self.w2v_encoder.load_state_dict(state_dict)

        out_channels = 112
        self.decoder = nn.Sequential(
            nn.Conv1d(encoder_hidden_dim, out_channels, 25, dilation=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, self.n_class, 1)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        output = self.w2v_encoder(x, features_only=True)
        output = output['x']
        b, t, c = output.shape
        output = output.reshape(b, c, t)
        output = self.decoder(output).squeeze()
        return output
        #if self.training:
            #return self.softmax(output)
        #else:
        #    return output

    def _get_name(self):
        return 'wav2keyword'
