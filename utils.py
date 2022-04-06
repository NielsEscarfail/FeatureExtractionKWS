# Copyright (C) 2021 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
#
# Author: Cristian Cioflan, ETH (cioflanc@iis.ee.ethz.ch)
# Modified by: Niels Escarfail, ETH (nescarfail@ethz.ch)


import os

import torch
from sklearn.metrics import confusion_matrix

import numpy as np
#import seaborn as sn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import yaml

# Device setup
from feature_extraction.lin_pred_coef import LPCProcessor
from feature_extraction.mel_freq_cep_coef import MFCCProcessor



def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("Torch version: ", torch.version.__version__)
    print("Device: ", device)
    return device


def npy_to_txt(layer_number, activations):
    # Saving the input

    if layer_number == -1:
        tmp = activations.reshape(-1)
        f = open('input.txt', "a")
        f.write('# input (shape [1, 49, 10]),\\\n')
        for elem in tmp:
            if elem < 0:
                f.write(str(256 + elem) + ",\\\n")
            else:
                f.write(str(elem) + ",\\\n")
        f.close()
    # Saving layers' activations
    else:
        tmp = activations.reshape(-1)
        f = open('out_layer' + str(layer_number) + '.txt', "a")
        f.write('layers.0.relu1 (shape [1, 25, 5, 64]),\\\n')  # Hardcoded, should be adapted for better understanding.
        for elem in tmp:
            if (elem < 0):
                f.write(str(256 + elem) + ",\\\n")
            else:
                f.write(str(elem) + ",\\\n")
        f.close()


def remove_txt():
    # Removing old activations and inputs

    directory = '.'
    files_in_directory = os.listdir(directory)
    filtered_files = [file for file in files_in_directory if
                      (file.startswith("out_layer") or file.startswith("input.txt"))]
    for file in filtered_files:
        path_to_file = os.path.join(directory, file)
        os.remove(path_to_file)


def conf_matrix(labels, predicted, training_parameters):
    # Plotting confusion matrix

    labels = labels.cpu()
    predicted = predicted.cpu()
    cm = confusion_matrix(labels, predicted)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, index=[i for i in ['silence', 'unknown'] + training_parameters['wanted_words']],
                         columns=[i for i in ['silence', 'unknown'] + training_parameters['wanted_words']])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()


def parameter_generation():
    # Data processing parameters
    with open("config.yaml", "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    config_data_proc_params = params['data_processing_parameters']
    config_training_parameters = params['training_parameters']

    # Importing existing parameters
    time_shift_ms = config_data_proc_params['time_shift_ms']
    sample_rate = config_data_proc_params['sample_rate']
    clip_duration_ms = config_data_proc_params['clip_duration_ms']

    window_size_ms = config_data_proc_params['window_size_ms']
    window_stride_ms = config_data_proc_params['window_stride_ms']

    # Data processing computations
    time_shift_samples = int((time_shift_ms * sample_rate) / 1000)
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)

    # Regroup information to give to model
    data_processing_parameters = {'feature_bin_count': config_data_proc_params['feature_bin_count'],
                                  'desired_samples': desired_samples, 'sample_rate': sample_rate,
                                  'spectrogram_length': spectrogram_length,
                                  'window_stride_samples': window_stride_samples,
                                  'window_size_samples': window_size_samples}

    training_parameters = {
        'data_dir': config_training_parameters['data_dir'],
        'data_url': config_training_parameters['data_url'],
        'epochs': config_training_parameters['epochs'],
        'batch_size': config_training_parameters['batch_size'],
        'silence_percentage': config_training_parameters['silence_percentage'],
        'unknown_percentage': config_training_parameters['unknown_percentage'],
        'validation_percentage': config_training_parameters['validation_percentage'],
        'testing_percentage': config_training_parameters['testing_percentage'],
        'background_frequency': config_training_parameters['background_frequency'],
        'background_volume': config_training_parameters['background_volume'],
    }

    target_words = config_training_parameters['target_words']
    wanted_words = target_words.split(',')
    wanted_words.pop()

    training_parameters['wanted_words'] = wanted_words
    training_parameters['time_shift_samples'] = time_shift_samples

    return training_parameters, data_processing_parameters


# TODO: parameter validation (model / feature extr compatible)
def create_model(model_name):
    if model_name == 'dscnn':
        from models.dscnn import DSCNN
        return DSCNN(use_bias=True)
    elif model_name == 'wav2vec':
        from models.wav2vec import Wav2Keyword
    elif model_name == 'bcresnet':
        from models.bcresnet import BCResNet

    else:
        raise NotImplementedError


def create_audioprocessor(ft_extr, training_parameters, data_processing_parameters):
    if ft_extr == 'mfcc':
        return MFCCProcessor(training_parameters, data_processing_parameters)
    elif ft_extr == 'lpc':
        return LPCProcessor(training_parameters, data_processing_parameters)
    else:
        raise NotImplementedError
