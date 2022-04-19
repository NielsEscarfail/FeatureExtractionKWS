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


import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import npy_to_txt


class DSCNN(torch.nn.Module):
    """DSCNN model Input shape:
    (1, 1, 49, 10) = (1, 1, spectrogram_length, feature_bin_count) for MFCC or
    (1, 1, 16000) for augmented data.
    """
    def __init__(self, model_params, use_bias=False):
        super(DSCNN, self).__init__()

        self.input_shape = model_params['model_input_shape']

        if self.input_shape == 16000:  # Shape [128, 1, 16000] for augmented data
            self.conv0 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(80, 1), stride=(8, 1))  # to [128, 4, 1991]
            self.bn0 = torch.nn.BatchNorm2d(4)
            self.relu0 = torch.nn.ReLU()

            self.conv0_0 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(40, 1), stride=(4, 1))  # to [128, 4, 1991]
            self.bn0_0 = torch.nn.BatchNorm2d(4)
            self.relu0_0 = torch.nn.ReLU()

            # self.reshape = torch.nn.Reshape()

            self.pad0_1 = nn.ConstantPad2d((11, 10, 0, 0), value=0.0)  # to [128, 4, 488, 22]
            self.conv0_1 = torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(20, 4), stride=(4, 2),
                                           bias=use_bias)  # to [128, 16, 118, 10]
            self.bn0_1 = torch.nn.BatchNorm2d(16)
            self.relu0_1 = torch.nn.ReLU()

            self.pad0_2 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)  # to [128, 16, 118, 12]
            self.conv0_2 = torch.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(20, 3), stride=(4, 2),
                                           bias=use_bias)  # to [128, 64, 25, 5]
        else:  # original implementation for MFCC data
            self.pad1 = nn.ConstantPad2d((1, 1, 5, 5), value=0.0)  # [128, 1, 49, 10] to [128, 1, 59, 12]
            self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(10, 4), stride=(2, 2),
                                         bias=use_bias)  # [128, 1, 59, 12] [128, 64, 25, 5]

        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU()
        # have first conv layer as adaptor -> takes input and makes sure it is always 25*5 then observe acc.
        # If results are bad, try to move away from 25*5 inter-shape and go larger
        self.pad2 = nn.ConstantPad2d((1, 1, 1, 1), value=0.)  # takes in torch.Size([128, 64, 25, 5])
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=64,
                                     bias=use_bias)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.relu3 = torch.nn.ReLU()

        self.pad4 = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=64,
                                     bias=use_bias)
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.relu4 = torch.nn.ReLU()
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn5 = torch.nn.BatchNorm2d(64)
        self.relu5 = torch.nn.ReLU()

        self.pad6 = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv6 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=64,
                                     bias=use_bias)
        self.bn6 = torch.nn.BatchNorm2d(64)
        self.relu6 = torch.nn.ReLU()
        self.conv7 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn7 = torch.nn.BatchNorm2d(64)
        self.relu7 = torch.nn.ReLU()

        self.pad8 = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv8 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=64,
                                     bias=use_bias)
        self.bn8 = torch.nn.BatchNorm2d(64)
        self.relu8 = torch.nn.ReLU()
        self.conv9 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn9 = torch.nn.BatchNorm2d(64)
        self.relu9 = torch.nn.ReLU()

        self.avg = torch.nn.AvgPool2d(kernel_size=(25, 5), stride=1)
        self.fc1 = torch.nn.Linear(64, 12, bias=use_bias)

    def forward(self, x, save=False):
        save = False
        if save:
            x = self.pad1(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            npy_to_txt(0, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))

            x = self.pad2(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            npy_to_txt(1, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            npy_to_txt(2, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))

            x = self.pad4(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            npy_to_txt(3, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))
            x = self.conv5(x)
            x = self.bn5(x)
            x = self.relu5(x)
            npy_to_txt(4, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))

            x = self.pad6(x)
            x = self.conv6(x)
            x = self.bn6(x)
            x = self.relu6(x)
            npy_to_txt(5, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))
            x = self.conv7(x)
            x = self.bn7(x)
            x = self.relu7(x)
            npy_to_txt(6, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))

            x = self.pad8(x)
            x = self.conv8(x)
            x = self.bn8(x)
            x = self.relu8(x)
            npy_to_txt(7, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))
            x = self.conv9(x)
            x = self.bn9(x)
            x = self.relu9(x)
            npy_to_txt(8, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))

            x = self.avg(x)
            npy_to_txt(9, x.int().cpu().detach().numpy())
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            npy_to_txt(10, x.int().cpu().detach().numpy())

        else:
            if self.input_shape == 16000:
                # For augmented data input without MFCC preprocessing.
                # reshapes the input from [batch_size, 1, 16000] to [batch_size, 64, 25, 5] for compatibility.
                x = self.conv0(x)
                x = self.bn0(x)
                x = self.relu0(x)

                x = self.conv0_0(x)
                x = self.bn0_0(x)
                x = self.relu0_0(x)

                # x = x[:, :, :, None]

                x = self.pad0_1(x)
                x = self.conv0_1(x)
                x = self.bn0_1(x)
                x = self.relu0_1(x)

                x = self.pad0_2(x)
                x = self.conv0_2(x)

            else:
                x = self.pad1(x)
                x = self.conv1(x)

            x = self.bn1(x)
            x = self.relu1(x)

            x = self.pad2(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)

            x = self.pad4(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            x = self.conv5(x)
            x = self.bn5(x)
            x = self.relu5(x)

            x = self.pad6(x)
            x = self.conv6(x)
            x = self.bn6(x)
            x = self.relu6(x)
            x = self.conv7(x)
            x = self.bn7(x)
            x = self.relu7(x)

            x = self.pad8(x)
            x = self.conv8(x)
            x = self.bn8(x)
            x = self.relu8(x)
            x = self.conv9(x)
            x = self.bn9(x)
            x = self.relu9(x)

            x = self.avg(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)

        return x

    def _get_name(self):
        return 'dscnn'
