import torch
from models.dscnn import DSCNN, DSCNNAVGPOOL, DSCNNSUBCONV
from models.kwt import KWT
import torchaudio
import fairseq

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

    elif model_name == 'wav2vec_pt10m':
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M
        return bundle.get_model()
        # return torchaudio.models.DeepSpeech(n_feature=1, n_hidden=8, n_class=12)

    elif model_name == 'wav2vec_small':
        from torchaudio.models.wav2vec2.utils import import_fairseq_model
        # Load model using fairseq
        model_file = '/Users/nielsescarfail/Desktop/FeatureExtractionKWS/models/wav2vec/wav2vec_small.pt'
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file])
        original = model[0]
        imported = import_fairseq_model(original)
        return imported

    elif model_name == 'mlp':
        from mlp_mixer_pytorch import MLPMixer
        return MLPMixer(
            image_size=model_params['model_input_shape'],
            channels=1,
            patch_size=1,
            dim=32,
            depth=3,
            num_classes=12
        )

    elif model_name == 'dscnn_avgpool':
        return DSCNNAVGPOOL(model_params, use_bias=True)

    if model_name == 'dscnn_subconv':
        return DSCNNSUBCONV(model_params, use_bias=True)

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
        from models.wav2vec import wav2keyword
        from models.wav2vec.wav2vec import generate_w2v_model_params
        trained_model = torch.load(model_params['pt'])
        w2v_model_config = vars(trained_model['args'])
        w2v_model_config = generate_w2v_model_params(w2v_model_config)
        state_dict = trained_model['model']
        model = wav2keyword.Wav2Keyword(cfg=w2v_model_config, state_dict=state_dict)
        return model

    else:
        raise NotImplementedError
