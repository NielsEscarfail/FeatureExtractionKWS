# Semester project: Feature Extraction for Speech Recognition

## Report
Please refer to the [full report](./report.pdf) for a detailed description of the project.

## Directory layout
Here, an overview over the files is given. The main components are the **dataset**, **architecture**, **losses** and **benchmarking**.
```
.
├── benchmark_models
├── dataset                   <--- Dataset files
├── once_for_all
│   ├── elastic_nn
│   │   ├── modules
│   │   │   ├── dynamic_layers.py 
│   │   │   └── dynamic_op.py
│   │   ├── networks
│   │   │   └── ofa_kws_net.py             <--- OFAKWSNet implementation
│   │   ├── training
│   │   │   └── progressive_shrinking.py   <--- PS for elastic feature extraction, training, evaluation
│   │   └── utils.py
│   ├── evaluation                          
│   │   ├── arch_encoder.py                <--- OFAKWSNet architecture encoder (untested)
│   │   └── perf_dataset.py                <--- Evaluatiom
│   ├── networks
│   │   └── kws_net.py                     <--- KWSNet implementation
│   └── run_manager
│       ├── data_provider.py               <--- Feature extraction
│       ├── dataset.py                     <--- Data querying + augmentation
│       ├── run_config.py                  <--- Configurations
│       └── run_manager.py                 <--- Run management
├── environment.yml
├── notebooks
├── README.md
├── run_all.sh              <--- Run all training phases then evaluate 1000 subnets
├── train_ofa_net.py        <--- Training script
└── eval_ofa_net.py         <--- Evaluation data-gathering script


### Setup
Create a new conda environment and install the packages from *environment.yml* (Linux):
```bash
conda env create -n kwsenv -f environment.yml
conda activate kwsenv
```

### Dataset
To train or evaluate the OFAKWSNet model, you first need to download the `speech-commands-v0.02` dataset.

### Training:
To train a model from scratch, create a feature extraction parameters configuration in the `config_utils.py` file, specifying a certain `params_id`, then run:

```
bash run_all.sh [feature_extraction_type] [params_id]
```

## References
[ONCE-FOR-ALL](https://github.com/mit-han-lab/once-for-all/tree/master)  
[KWS-ON-PULP](https://github.com/pulp-platform/kws-on-pulp)