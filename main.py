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
from copy import deepcopy
import nemo
import torch
from pthflops import count_ops
from torchsummary import summary
from train import Trainer
from utils import remove_txt, parameter_generation, setup_device
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

    # Get input shape # TODO
    model_input_shape = model_parameters['model_input_shape']


    # Model analysis # TODO: input variability for other inputs than MFCC input
    model.to(device)
    summary(model, (1, 49, data_processing_parameters['feature_bin_count']))
    # summary(model, (1, 16000, 1))
    dummy_input = torch.rand(1, 1, 49, data_processing_parameters['feature_bin_count']).to(device)
    # dummy_input = torch.rand(1, 1, 16000, 1).to(device)
    count_ops(model, dummy_input)

    # Training initialization
    training_environment = Trainer(audio_processor, training_parameters, model, device)

    # Removing stored inputs and activations
    remove_txt()

    if args.load_trained:  # If --load_pretrained_model True, ignore training and load pretrained model
        if torch.cuda.is_available():
            model.load_state_dict(
                torch.load(os.path.join(model_save_dir, 'model.pth', map_location=torch.device('cuda'))))
        else:
            model.load_state_dict(torch.load(os.path.join(model_save_dir, 'model.pth')))
        print("Loaded model from: ", model_save_dir)

    else:  # Train and save the model
        start = time.time()
        training_environment.train(model=model, save_path=model_save_dir)
        print('Finished Training on GPU in {:.2f} seconds'.format(time.time() - start))

    # Quantization and validation phase
    print("\nPytorch implementation accuracy:")
    acc = training_environment.validate(model=model, mode='testing', batch_size=-1)

    # Initiating quantization process: making the model quantization aware
    quantized_model = nemo.transform.quantize_pact(deepcopy(model), dummy_input=torch.randn((1, 1, 49, 10)).to(device))
    # quantized_model = nemo.transform.quantize_pact(deepcopy(model), dummy_input=torch.randn((1, 1, 16000, 1)).to(device))

    precision_8 = {
        "conv1": {"W_bits": 7},
        "relu1": {"x_bits": 8},
        "conv2": {"W_bits": 7},
        "relu2": {"x_bits": 8},
        "conv3": {"W_bits": 7},
        "relu3": {"x_bits": 8},
        "conv4": {"W_bits": 7},
        "relu4": {"x_bits": 8},
        "conv5": {"W_bits": 7},
        "relu5": {"x_bits": 8},
        "conv6": {"W_bits": 7},
        "relu6": {"x_bits": 8},
        "conv7": {"W_bits": 7},
        "relu7": {"x_bits": 8},
        "conv8": {"W_bits": 7},
        "relu8": {"x_bits": 8},
        "conv9": {"W_bits": 7},
        "relu9": {"x_bits": 8},
        "fc1": {"W_bits": 7}
    }
    quantized_model.change_precision(bits=1, min_prec_dict=precision_8, scale_weights=True, scale_activations=True)

    # Calibrating model's scaling by collecting largest activations
    with quantized_model.statistics_act():
        training_environment.validate(model=quantized_model, mode='validation', batch_size=128)
    quantized_model.reset_alpha_act()

    # Remove biases after FQ stage
    quantized_model.remove_bias()

    print("\nFakeQuantized @ 8b accuracy (calibrated):")
    acc = training_environment.validate(model=quantized_model, mode='testing', batch_size=-1)

    quantized_model.qd_stage(eps_in=255. / 255)  # The activations are already in 0-255

    print("\nQuantizedDeployable @ mixed-precision accuracy:")
    acc = training_environment.validate(model=quantized_model, mode='testing', batch_size=-1)

    quantized_model.id_stage()

    print("\nIntegerDeployable @ mixed-precision accuracy:")
    acc = training_environment.validate(model=quantized_model, mode='testing', batch_size=-1, integer=True)

    # Saving the model TODO rewrite path
    nemo.utils.export_onnx(model_save_dir + '/model.onnx', quantized_model, quantized_model, (1, 49, 10))

    # Saving the activations for comparison within Dory
    acc = training_environment.validate(model=quantized_model, mode='testing', batch_size=1, integer=True, save=True)
