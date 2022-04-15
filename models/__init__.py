import torch
from models.wav2vec import wav2keyword
from models.wav2vec.wav2vec import generate_w2v_model_params
from models.dscnn import DSCNN


def create_model(model_name, model_params):
    """
    Instantiates the given model with the given model parameters.
    Args:
      model_name: Which model to instantiate.
      model_params: Model parameters, described at config.yaml

    Returns:
      Model to perform KWS
    """
    if model_name == 'dscnn':
        return DSCNN(model_params, use_bias=True)

    elif model_name == 'wav2vec':
        trained_model = torch.load(model_params['pt'])
        w2v_model_config = vars(trained_model['args'])
        w2v_model_config = generate_w2v_model_params(w2v_model_config)
        state_dict = trained_model['model']
        model = wav2keyword.Wav2Keyword(cfg=w2v_model_config, state_dict=state_dict)
        return model

    else:
        raise NotImplementedError
