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
import shutil
import time
import torch
import torch.nn.functional as F
from feature_extraction import dataset
from utils import conf_matrix, npy_to_txt


class Trainer:
    def __init__(self, audio_processor, training_parameters, model, device):
        self.audio_processor = audio_processor
        self.training_parameters = training_parameters
        self.model = model
        self.device = device

        # Training hyperparameters, set in config.yaml
        if training_parameters['criterion'] == 'CrossEnt':
            self.criterion = torch.nn.CrossEntropyLoss().cuda()

        if training_parameters['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=training_parameters['initial_lr'])

        lambda_lr = lambda epoch: 1 if epoch < 15 else 1 / 5 if epoch < 25 else 1 / 10 if epoch < 35 else 1 / 20
        if training_parameters['scheduler'] == 'LambdaLR':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda_lr)
        elif training_parameters['scheduler'] == 'ReduceLROnPlateau':
            self.metric = 0  # used for learning rate policy 'plateau'
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2,
                                                                        patience=5, verbose=True)

        if self.model._get_name() == 'wav2keyword':
            self.optimizer = torch.optim.Adam([
                {'params': model.w2v_encoder.parameters(), 'lr': 1e-5},
                {'params': model.decoder.parameters(), 'lr': 5e-4},
            ], weight_decay=1e-5)

        # Save the feature extraction method run times in training and validation mode
        self.dt_train = self.time_get_data(mode='training')
        self.dt_val = self.time_get_data(mode='validation')

    def time_get_data(self, mode):
        """Computes the average time to gather a sample with the given feature extraction method."""
        start = time.time()
        data = dataset.AudioGenerator(mode, self.audio_processor, self.training_parameters)

        n_batches = 20  # Increase for better approximation, low for performance reasons
        for i in range(n_batches):
            inputs, labels = data[0]
        dt = (time.time() - start)
        dt_batch = dt / n_batches

        print('\nFeature extraction method: {}'.format(self.audio_processor.feature_extraction_method))
        print('Dataset shape: inputs: {}, labels: {}'.format(inputs.shape, labels.shape))
        print('Time to read {} training batches: {:.2f} s.  //  {:.4f} seconds per training batch.'
              .format(n_batches, dt, dt_batch))

        return dt_batch

    def validate(self, model=None, mode='validation', batch_size=-1, statistics=False, integer=False, save=False):
        """Validate the model."""

        training_parameters = self.training_parameters
        training_parameters['batch_size'] = batch_size
        data = dataset.AudioGenerator(mode, self.audio_processor, training_parameters)
        model.eval()

        correct = 0
        total = 0
        start = time.time()

        with torch.no_grad():
            inputs, labels = data[0]

            # Set inputs depending on the model
            if self.model._get_name() == 'wav2keyword':
                inputs = torch.Tensor(inputs).to(self.device)

            elif self.model._get_name() in {'dscnn_subconv', 'dscnn_maxpool'}:
                inputs = torch.Tensor(inputs[:, None, :]).to(self.device)

            elif self.model._get_name() in {'dscnn'}:
                # Set inputs depending on the model + feature extraction method used
                if self.audio_processor.feature_extraction_method in {'mfcc', 'mel_spectrogram'}:
                    inputs = torch.Tensor(inputs[:, None, :, :]).to(self.device)
                elif self.audio_processor.feature_extraction_method in {'augmented', 'raw', 'dwt'}:
                    inputs = torch.Tensor(inputs[:, None, :, None]).to(self.device)

            elif self.model._get_name() == 'kwt':
                inputs = torch.Tensor(inputs).to(self.device)

            else:
                inputs = torch.Tensor(inputs[:, :]).to(self.device)

            labels = torch.Tensor(labels).long().to(self.device)
            model = model.to(self.device)

            if integer:
                model = model.cpu()
                inputs = inputs * 255. / 255
                inputs = inputs.type(torch.uint8).type(torch.float).cpu()

            if save:
                model = model.cpu()
                inputs = inputs.type(torch.uint8).type(torch.float).cpu()
                outputs = F.softmax(model(inputs, save), dim=1)
                outputs = outputs.to(self.device)
                npy_to_txt(-1, inputs.int().cpu().detach().numpy())
            else:
                outputs = F.softmax(model(inputs), dim=1)
                outputs = outputs.to(self.device)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if statistics:
                conf_matrix(labels, predicted, self.training_parameters)

        total_time = (time.time() - start)
        print('Accuracy of the network on the %s set: %.2f %%' % (mode, 100 * correct / total))
        print('Took %.4f seconds for %.f samples, %.6f seconds per sample.' % (total_time, total, total_time / total))
        return 100 * correct / total, total_time / total

    def train(self, model, save_path):
        """Train the model."""

        best_acc = 0
        for epoch in range(0, self.training_parameters['epochs']):

            print("Epoch: " + str(epoch + 1) + "/" + str(self.training_parameters['epochs']))
            data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters)
            model.train()

            running_loss = 0.0
            total = 0
            correct = 0

            for minibatch in range(len(data)):

                inputs, labels = data[0]  # Returns a random index anyway

                # Set inputs depending on the model


                if self.model._get_name() == 'wav2keyword':
                    inputs = torch.Tensor(inputs).to(self.device)

                elif self.model._get_name() in {'dscnn_subconv', 'dscnn_avgpool', 'dscnn_maxpool'}:
                    inputs = torch.Tensor(inputs[:, None, :]).to(self.device)

                elif self.model._get_name() in {'dscnn', 'mlp'}:
                    # Set inputs depending on the model + feature extraction method used
                    if self.audio_processor.feature_extraction_method in {'mfcc', 'mel_spectrogram'}:
                        inputs = torch.Tensor(inputs[:, None, :, :]).to(self.device)
                    elif self.audio_processor.feature_extraction_method in {'augmented', 'raw', 'dwt'}:
                        inputs = torch.Tensor(inputs[:, None, :, None]).to(self.device)

                elif self.model._get_name() == 'kwt':
                    inputs = torch.Tensor(inputs).to(self.device)

                else:
                    inputs = torch.Tensor(inputs).to(self.device)

                labels = torch.Tensor(labels).to(self.device).long()

                # Zero out the parameter gradients after each mini-batch
                self.optimizer.zero_grad()

                # Train, compute loss, update optimizer
                model = model.to(self.device)
                outputs = F.softmax(model(inputs), dim=1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Compute training statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Print information every 20 minibatches
                if minibatch % 20 == 0:
                    print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (
                        minibatch + 1, len(data), running_loss / 10, 100 * correct / total))
                    running_loss = 0.0

            tmp_acc, _ = self.validate(model, 'validation', 256)  # used to be 128

            # Save best performing network
            if tmp_acc > best_acc:
                best_acc = tmp_acc
                PATH = './model_acc_' + str(best_acc) + '.pth'
                PATH = os.path.join(save_path, PATH)
                torch.save(model.state_dict(), PATH)

        # Update scheduler
        if self.training_parameters['scheduler'] == 'ReduceLROnPlateau':  # ReduceLROnPlateau requires metrics param
            self.scheduler.step(self.metric)
        else:
            self.scheduler.step()

        # Save model state dict
        PATH = os.path.join(save_path, 'model.pth')
        torch.save(model.state_dict(), PATH)

        # Save a copy of the config file
        shutil.copy('config.yaml', save_path)
