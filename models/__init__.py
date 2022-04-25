import torch
from models.wav2vec import wav2keyword
from models.wav2vec.wav2vec import generate_w2v_model_params
from models.dscnn import DSCNN
from models.kwt import KWT


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

    elif model_name == 'kwt':
        return KWT(img_x=model_params['img_x'], img_y=model_params['img_y'],
                   patch_x=model_params['patch_x'], patch_y=model_params['patch_y'],
                   num_classes=model_params['num_classes'],
                   dim=model_params['dim'],
                   depth=model_params['depth'],
                   heads=model_params['heads'],
                   mlp_dim=model_params['mlp_dim'],
                   pool=model_params['pool'],
                   channels=model_params['channels'],
                   dim_head=model_params['dim_head'],
                   dropout=model_params['dropout'],
                   emb_dropout=model_params['emb_dropout'])

    elif model_name == 'wav2vec':
        trained_model = torch.load(model_params['pt'])
        w2v_model_config = vars(trained_model['args'])
        w2v_model_config = generate_w2v_model_params(w2v_model_config)
        state_dict = trained_model['model']
        model = wav2keyword.Wav2Keyword(cfg=w2v_model_config, state_dict=state_dict)
        return model

    else:
        raise NotImplementedError
