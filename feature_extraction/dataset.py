# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# Adapted by: Cristian Cioflan, ETH (cioflanc@iis.ee.ethz.ch)
# Modified by: Niels Escarfail, ETH (nescarfail@ethz.ch)


import glob
import hashlib
import math
import os
import os.path
import random
import re

import numpy as np
import soundfile as sf
import tensorflow as tf
import torch
import torchaudio

MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M
BACKGROUND_NOISE_LABEL = '_background_noise_'
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
RANDOM_SEED = 59185


def prepare_words_list(wanted_words):
    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def which_set(filename, validation_percentage, testing_percentage):
    # Split dataset in training, validation, and testing set
    # Should be modified to load validation data from validation_list.txt
    # Should be modified to load testing data from testing_list.txt

    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name.encode()).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result


class AudioProcessor(object):
    def __init__(self, training_parameters, data_processing_parameters):

        self.data_directory = training_parameters['data_dir']
        self.generate_background_noise()
        self.generate_data_dictionary(training_parameters)
        self.data_processing_parameters = data_processing_parameters

    def generate_data_dictionary(self, training_parameters):
        # For each data set, generate a dictionary containing the path to each file, its label, and its speaker.
        # Make sure the shuffling and picking of unknowns is deterministic.
        random.seed(RANDOM_SEED)
        wanted_words_index = {}

        for index, wanted_word in enumerate(training_parameters['wanted_words']):
            wanted_words_index[wanted_word] = index + 2

        # Prepare data sets
        self.data_set = {'validation': [], 'testing': [], 'training': []}
        unknown_set = {'validation': [], 'testing': [], 'training': []}
        all_words = {}
        # Find all audio samples
        search_path = os.path.join(self.data_directory, '*', '*.wav')

        for wav_path in glob.glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            speaker_id = wav_path.split('/')[3].split('_')[0]  # Hardcoded, should use regex.
            word = word.lower()

            # Ignore background noise, as it has been handled by generate_background_noise()
            if word == BACKGROUND_NOISE_LABEL:
                continue

            all_words[word] = True
            # Determine the set to which the word should belong
            set_index = which_set(wav_path, training_parameters['validation_percentage'],
                                  training_parameters['testing_percentage'])

            # If it's a known class, store its detail, otherwise add it to the list
            # we'll use to train the unknown label.
            # If we use 35 classes - all are known, hence no unkown samples
            if word in wanted_words_index:
                self.data_set[set_index].append({'label': word, 'file': wav_path, 'speaker': speaker_id})
            else:
                unknown_set[set_index].append({'label': word, 'file': wav_path, 'speaker': speaker_id})

        if not all_words:
            raise Exception('No .wavs found at ' + search_path)
        for index, wanted_word in enumerate(training_parameters['wanted_words']):
            if wanted_word not in all_words:
                raise Exception('Expected to find ' + wanted_word +
                                ' in labels but only found ' +
                                ', '.join(all_words.keys()))

        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_set['training'][0]['file']

        # Add silence and unknown words to each set
        for set_index in ['validation', 'testing', 'training']:

            set_size = len(self.data_set[set_index])
            silence_size = int(math.ceil(set_size * training_parameters['silence_percentage'] / 100))
            for _ in range(silence_size):
                self.data_set[set_index].append({
                    'label': SILENCE_LABEL,
                    'file': silence_wav_path,
                    'speaker': "None"
                })

            # Pick some unknowns to add to each partition of the data set.
            random.shuffle(unknown_set[set_index])
            unknown_size = int(math.ceil(set_size * training_parameters['unknown_percentage'] / 100))
            self.data_set[set_index].extend(unknown_set[set_index][:unknown_size])

        # Make sure the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_set[set_index])

        # Prepare the rest of the result data structure.
        self.words_list = prepare_words_list(training_parameters['wanted_words'])
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
            else:
                self.word_to_index[word] = UNKNOWN_WORD_INDEX
        self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

    def generate_background_noise(self):
        # Load background noise, used to augment clean speech

        self.background_noise = []
        background_dir = os.path.join(self.data_directory, BACKGROUND_NOISE_LABEL)
        if not os.path.exists(background_dir):
            return self.background_noise

        search_path = os.path.join(self.data_directory, BACKGROUND_NOISE_LABEL, '*.wav')
        for wav_path in glob.glob(search_path):
            # List of tensor, each one is a background noise
            sf_loader, _ = sf.read(wav_path)
            wav_file = torch.Tensor(np.array([sf_loader]))
            self.background_noise.append(wav_file[0])

        if not self.background_noise:
            raise Exception('No background wav files were found in ' + search_path)

    def get_size(self, mode):
        # Compute data set size

        return len(self.data_set[mode])

    def get_data(self, mode, training_parameters):
        return NotImplementedError


class AudioGenerator(torch.utils.data.Dataset):
    # Returning batches of data (MFCCs) and labels

    def __init__(self, mode, audio_processor, training_parameters):
        self.mode = mode
        self.audio_processor = audio_processor
        if self.mode != 'training':
            training_parameters['background_frequency'] = 0
            training_parameters['background_volume'] = 0
            training_parameters['time_shift_samples'] = 0
        self.training_parameters = training_parameters

    def __len__(self):
        # Return dataset length

        if self.training_parameters['batch_size'] == -1:
            return (len(self.audio_processor.data_set[self.mode]))
        else:
            return int(len(self.audio_processor.data_set[self.mode]) / self.training_parameters['batch_size'])

    def __getitem__(self, idx):
        # Return a random batch of data, unless training_parameters['batch_size'] == -1

        data, labels = self.audio_processor.get_data(self.mode, self.training_parameters)

        return data, labels
