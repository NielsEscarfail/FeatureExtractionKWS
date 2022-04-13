# Semester project: Feature Extraction for Speech Recognition
### In progress:
- Wav2Vec / BCResNet implementations
- Mel spectrogram features

### Training KWS models
To train then test a model, run:
```
python main.py --model model_name --ft_extr feature_extraction_method 
```

For instance, to run a DSCNN model with MFCC feature extraction, run:
```
python main.py --model dscnn --ft_extr mfcc
```

To load and test a model run: (--model_save_dir must be in models/trained_models)
```
python main.py --load_trained --model_save_dir dscnn_mfcc
```

### Abstract
The goals of this project are to:
- Modify the [kws-on-pulp](https://github.com/pulp-platform/kws-on-pulp) framework to enable running and comparison of existing state-of-the-art kws feature extraction methods and model architectures, 
- Analyse and compare the performance of existing KWS models, more specifically their feature extraction methods.
- Explore new feature extraction methods for KWS to obtain a high-accuracy, low latency and power consumption model.

