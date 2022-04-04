import torch
from torch import nn
# import fairseq
# from wav2vec import Wav2VecModel
#from torchaudio.models import Wav2Vec2Model
# from fairseq import models

from .fairseq.dataclass.utils import convert_namespace_to_omegaconf
from .fairseq import tasks
"""
W2V_PRETRAINED_MODEL_PATH = "models/wav2vec/wav2vec_small.pt"
state_dict = torch.load(W2V_PRETRAINED_MODEL_PATH)

cfg = convert_namespace_to_omegaconf(state_dict['args'])

task = tasks.setup_task(cfg.task)
w2v_encoder = task.build_model(cfg.model)
"""
class Wav2Keyword(nn.Module):
    def __init__(self, n_class=12, encoder_hidden_dim=768, cfg=None, state_dict=None, task=None):
        super(Wav2Keyword, self).__init__()
        self.n_class = n_class
        assert not cfg is None
        assert not state_dict is None
        assert not task is None

        self.w2v_encoder = task.build_model(cfg.model)
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
        output = self.w2v_encoder(**x, features_only=True)
        output = output['x']
        b, t, c = output.shape
        output = output.reshape(b, c, t)
        output = self.decoder(output).squeeze()
        if self.training:
            return self.softmax(output)
        else:
            return output

