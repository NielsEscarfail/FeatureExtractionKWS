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
import argparse
import os
import time
import torch
from train import Trainer
from utils import remove_txt, parameter_generation, setup_device, export_all_results_to_csv
from models import create_model
from feature_extraction.dataset import AudioProcessor

if __name__ == '__main__':
    device = setup_device()  # Set up cuda device

    # Gather arguments
    parser = argparse.ArgumentParser("KWS model trainer")
    parser.add_argument("--model", default="dscnn", help="Model to perform kws", type=str)
    parser.add_argument("--ft_extr", default="mfcc", help="Feature extraction method", type=str)
    parser.add_argument("--model_save_dir", default=None,
                        help="Directory name where the trained model will be saved, or is stored",
                        type=str)
    parser.add_argument("--load_trained", default=False, action='store_true',
                        help="If True, load an already trained model from models/trained_models/model_save_dir")

    args = parser.parse_args()
    model_save_dir = args.model_save_dir

    # Manage model save directory TODO: move to utils
    if model_save_dir is None:
        model_save_dir = os.getcwd() + "/models/trained_models/" + args.model + "_" + args.ft_extr
        print("setting model save directory to: ", model_save_dir)
    else:
        model_save_dir = os.path.join(os.getcwd(), "/models/trained_models/", model_save_dir)
    # Create target directory & all intermediate directories if they don't exist
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        print("Created directory: ", model_save_dir)

    # Parameter generation
    training_parameters, data_processing_parameters, model_parameters = parameter_generation(args.model, args.ft_extr)

    print("training_parameters : ", training_parameters)
    print("data_processing_parameters : ", data_processing_parameters)
    print("model_parameters : ", model_parameters)

    # Instantiate model and feature extraction method based on --model and --ft_extr args
    model = create_model(args.model, model_parameters)
    audio_processor = AudioProcessor(training_parameters, data_processing_parameters)

    # Train/test/validation set split
    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')

    # Get input shape # TODO cleaner or move to utils
    if data_processing_parameters['feature_extraction_method'] in {'mfcc', 'mel_spectrogram'}:
        model_input_shape_summary = (1, 49, data_processing_parameters['feature_bin_count'])
        model_input_shape = (1, 1, 49, data_processing_parameters['feature_bin_count'])

    elif data_processing_parameters['feature_extraction_method'] == 'kwt':
        model_input_shape_summary = (49, 40)
        model_input_shape = (1, 49, 40)

    elif args.model in {'dcsnn_maxpool'}:
        model_input_shape_summary = (1, 16000)
        model_input_shape = (1, 1, 16000)

    else:  # raw, augmented, (dwr for now)
        model_input_shape_summary = (1, 16000, 1)
        model_input_shape = (1, 1, 16000, 1)

    # Model analysis
    model.to(device)
    # summary(model, model_input_shape_summary)
    dummy_input = torch.rand(model_input_shape).to(device)
    # count_ops(model, dummy_input)

    # Training initialization
    training_environment = Trainer(audio_processor, training_parameters, model, device)

    # Removing stored inputs and activations
    remove_txt()
    training_time = 0

    if args.load_trained:  # If --load_trained, ignore training and load pretrained model
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(os.path.join(model_save_dir, 'model.pth'), map_location=torch.device('cuda')))
        else:
            model.load_state_dict(torch.load(os.path.join(model_save_dir, 'model.pth')))
        print("\nLoaded model from: ", model_save_dir)

    else:  # Train and save the model
        start = time.time()
        training_environment.train(model=model, save_path=model_save_dir)
        training_time = time.time() - start
        print('\nFinished Training on GPU in {:.2f} seconds'.format(training_time))

    # Save all results and export to csv  TODO add model size
    test_acc, run_test_sample_time = training_environment.validate(model=model, mode='testing', batch_size=-1)
    val_acc, run_val_sample_time = training_environment.validate(model=model, mode='validation', batch_size=-1)

    results = {'test_acc': test_acc, 'val_acc': val_acc,
               'run_test_sample_time': run_test_sample_time, 'run_val_sample_time': run_val_sample_time,
               'time_get_train_batch': training_environment.dt_train, 'time_get_val_batch': training_environment.dt_val}

    export_all_results_to_csv(model_save_dir, training_environment, training_parameters, data_processing_parameters,
                              model_parameters, results)
