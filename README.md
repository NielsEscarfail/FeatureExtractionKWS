# Semester project: Feature Extraction for Speech Recognition
## In progress:
- Parameters saving (in models/trained_model/trained_model_name)
- LPC implementation: https://librosa.org/doc/main/generated/librosa.lpc.html
- Wav2Vec / BCResNet / CRNN / Transformer input modulation (will prioritize advancing on feature extraction methods first)

## Abstract
The goals of this project are to:
- Modify the [kws-on-pulp](https://github.com/pulp-platform/kws-on-pulp) framework to enable running and comparison of existing state-of-the-art kws feature extraction methods and model architectures, 
- Analyse and compare the performance of existing KWS models, more specifically their feature extraction methods.
- Explore new feature extraction methods for KWS to obtain a high-accuracy, low latency and power consumption model.

## Training KWS models
To train then test a model, run:
python main.py --model model_name --ft_extr mfcc 

To load and test a model run: (--model_save_dir must be in models/trained_models, you can also just specify the model and feature extraction method)
python main.py --load_trained True --model_save_dir dscnn_mfcc