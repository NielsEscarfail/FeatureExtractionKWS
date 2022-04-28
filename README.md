# Semester project: Feature Extraction for Speech Recognition
### In progress:
- Models:
  - Wav2Vec - in testing
  - DSCNN - variations to first layer to enable input modularisation
  - BCResNet - implementing
  
- Feature extraction:
  - Mel spectrogram features
  
- Results gathering code implementation.
- Feature extraction methods plotting.

### Tested + Supported model / feature extraction method pairs:
- No regards to accuracy so far. 
- Best performing is wav2vec raw, but needs low batch size (<16), and dscnn mfcc.
```
python main.py --model dscnn --ft_extr mfcc
python main.py --model dscnn --ft_extr raw
python main.py --model dscnn --ft_extr augmented
python main.py --model dscnn --ft_extr dwt
python main.py --model dscnn_avgpool --ft_extr raw
python main.py --model dscnn_avgpool --ft_extr augmented
python main.py --model wav2vec --ft_extr raw
python main.py --model wav2vec --ft_extr augmented
```

### Running from Sassauna:
The project is accessible at: sassauna[0-4]/sem22f41/FeatureExtractionKWS
From there you can run:
```
bash
conda env activate kwsenv
pip install fairseq
python main.py --load_trained
```
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

