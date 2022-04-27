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
# import seaborn as sn
import pandas as pd
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import yaml


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


def parameter_generation(model, ft_extr):
    # Import config from config.yaml file
    with open("config.yaml", "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Initialise all parameter types
    config_data_proc_params = params['data_processing_parameters']
    config_training_parameters = params['training_parameters']
    model_parameters = params['model_parameters']

    # Add command-line chosen model and feature extraction method to parameters
    model_parameters['model_name'] = model
    config_data_proc_params['feature_extraction_method'] = ft_extr

    # Importing existing parameters
    time_shift_ms = config_data_proc_params['time_shift_ms']
    sample_rate = config_data_proc_params['sample_rate']
    clip_duration_ms = config_data_proc_params['clip_duration_ms']
    window_size_ms = config_data_proc_params['window_size_ms']
    window_stride_ms = config_data_proc_params['window_stride_ms']

    # Define model input shape depending on the feature extraction method used
    if ft_extr in {'raw', 'augmented', 'dwt'}:
        model_parameters['model_input_shape'] = sample_rate
    elif ft_extr in {'mfcc', 'mel_spectrogram'}:
        model_parameters['model_input_shape'] = {49, config_data_proc_params['feature_bin_count']}
    else:  # default
        model_parameters['model_input_shape'] = sample_rate

    # Add model parameters to pass depending on the model
    if model == 'kwt':  # might move this to argsparse
        model_parameters['img_x'] = 40
        model_parameters['img_y'] = 98
        model_parameters['patch_x'] = 40
        model_parameters['patch_y'] = 1
        model_parameters['num_classes'] = 12
        model_parameters['dim'] = 64
        model_parameters['depth'] = 2
        model_parameters['heads'] = 1
        model_parameters['mlp_dim'] = 256
        model_parameters['pool'] = 'cls'
        model_parameters['channels'] = 1
        model_parameters['dim_head'] = 64
        model_parameters['dropout'] = 0.
        model_parameters['emb_dropout'] = 0.

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

    data_processing_parameters = {'feature_bin_count': config_data_proc_params['feature_bin_count'],
                                  'desired_samples': desired_samples, 'sample_rate': sample_rate,
                                  'spectrogram_length': spectrogram_length,
                                  'window_stride_samples': window_stride_samples,
                                  'window_size_samples': window_size_samples,
                                  'feature_extraction_method': config_data_proc_params['feature_extraction_method']}

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
        'criterion': config_training_parameters['criterion'],
        'initial_lr': config_training_parameters['initial_lr'],
        'optimizer': config_training_parameters['optimizer'],
        'scheduler': config_training_parameters['scheduler'],
    }

    target_words = config_training_parameters['target_words']
    wanted_words = target_words.split(',')
    wanted_words.pop()

    training_parameters['wanted_words'] = wanted_words
    training_parameters['time_shift_samples'] = time_shift_samples

    return training_parameters, data_processing_parameters, model_parameters


def export_all_results_to_csv(model_save_dir, training_environment, training_parameters, data_processing_parameters, model_parameters, results):
    """ Exports all parameters, and resulting metrics of a run to trained_models/model_save_dir/results.csv"""

    # Remove unwanted data
    del training_parameters['data_dir']
    del training_parameters['data_url']
    del training_parameters['wanted_words']
    del model_parameters['pt']

    # Convert incompatible types to str
    model_parameters['model_input_shape'] = str(model_parameters['model_input_shape'])

    # Prepare the data from training/data/model parameters and results  TODO could use update to be cleaner
    cols = list(training_parameters.keys()) + list(data_processing_parameters.keys()) + list(model_parameters.keys()) + list(results.keys())
    vals = list(training_parameters.values()) + list(data_processing_parameters.values()) + list(model_parameters.values()) + list(results.values())
    data = dict(zip(cols, vals))

    # Create the df and export to csv
    df = pd.DataFrame({k: [v] for k, v in data.items()})

    print("Exporting: ", df.columns)
    print(df)

    df.to_csv(os.path.join(model_save_dir, 'results.csv'), index=False)
