# TODO: parameter validation (model / feature extr compatible)
import torch
from models.wav2vec.Wav2Keyword.fairseq import tasks
from models.wav2vec.Wav2Keyword.fairseq.dataclass.utils import convert_namespace_to_omegaconf
from models.wav2vec.wav2vec import generate_w2v_model_params
from models.wav2vec import wav2vec, wav2keyword


def create_model(model_name, model_params):
    if model_name == 'dscnn':
        from models.dscnn import DSCNN
        return DSCNN(use_bias=True)
    elif model_name == 'wav2vec':
        from models.wav2vec import Wav2Keyword
        trained_model = torch.load(model_params['pt'])
        w2v_model_config = vars(trained_model['args'])
        w2v_model_config = generate_w2v_model_params(w2v_model_config)
        print(w2v_model_config)
        state_dict = trained_model['model']
        model = wav2keyword.Wav2Keyword(cfg=w2v_model_config, state_dict=state_dict)
        return model
    else:
        raise NotImplementedError
