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

__all__ = ["AudioProcessor", "AudioGenerator"]

import glob
import hashlib
import itertools
import math
import os
import os.path
import random
import re
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchaudio.transforms import MFCC
import librosa
import time
import pywt

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
    """ Split dataset in training, validation, and testing set.
    Args:
        filename: File path of the data sample.
        validation_percentage: Percentage of the data set to use for validation.
        testing_percentage: Percentage of the data set to use for testing.
    Returns:
      One of 'training', 'testing' or 'validation'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when deciding which set to put a wav in,
    # so the data set creator has a way of grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should go into the training, testing,
    # or validation sets, and we want to keep existing files in the same set even if more files are subsequently
    # added. To do that, we need a stable way of deciding based on just the file name itself, so we do a hash of that
    # and then use that to generate a probability value that we use to assign it.
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
    """Handles the loading, partitioning, and feature extraction of audio training data."""

    def __init__(self,
                 feature_extraction_method="mfcc",
                 # Data processing params
                 time_shift_ms=200.0,
                 sample_rate=16000,
                 clip_duration_ms=1000,
                 window_size_ms=40.0,
                 window_stride_ms=20.0,
                 silence_percentage=10.0,
                 unknown_percentage=10.0,
                 validation_percentage=10.0,
                 testing_percentage=10.0,
                 background_frequency=0.8,
                 background_volume=0.2,
                 # Other params
                 data_dir='dataset/speech_commands_v0.02',
                 data_url='https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
                 target_words='yes,no,up,down,left,right,on,off,stop,go',  # GSCv2 - 12 words
                 # In testing
                 batch_size=-1,
                 ):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.feature_extraction_method = feature_extraction_method

        self.silence_percentage = silence_percentage
        self.unknown_percentage = unknown_percentage
        self.validation_percentage = validation_percentage
        self.testing_percentage = testing_percentage
        self.background_frequency = background_frequency
        self.background_volume = background_volume

        self.data_directory = data_dir

        # Data processing computed parameters
        self.time_shift_samples = int((time_shift_ms * sample_rate) / 1000)
        self.desired_samples = int(sample_rate * clip_duration_ms / 1000)

        # MFCC params
        self.window_size_samples = int(sample_rate * window_size_ms / 1000)
        self.window_stride_samples = int(sample_rate * window_stride_ms / 1000)
        length_minus_window = (self.desired_samples - self.window_size_samples)

        """if length_minus_window < 0:
            spectrogram_length = 0
        else:
            spectrogram_length = 1 + int(length_minus_window / self.window_stride_samples)
        prints 49
        """
        self.batch_size = batch_size

        self.wanted_words = target_words.split(',')
        self.wanted_words.pop()

        self.generate_background_noise()
        self.generate_data_dictionary()

    def generate_data_dictionary(self):
        """ For each data set, generate a dictionary containing the path to each file, its label, and its speaker.
        Args:
            training_parameters: data and model parameters, described at config.yaml.
        """

        # Make sure the shuffling and picking of unknowns is deterministic.
        random.seed(RANDOM_SEED)
        wanted_words_index = {}

        for index, wanted_word in enumerate(self.wanted_words):
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
            set_index = which_set(wav_path, self.validation_percentage, self.testing_percentage)

            # If it's a known class, store its detail, otherwise add it to the list
            # we'll use to train the unknown label.
            # If we use 35 classes - all are known, hence no unkown samples
            if word in wanted_words_index:
                self.data_set[set_index].append({'label': word, 'file': wav_path, 'speaker': speaker_id})
            else:
                unknown_set[set_index].append({'label': word, 'file': wav_path, 'speaker': speaker_id})

        if not all_words:
            raise Exception('No .wavs found at ' + search_path)
        for index, wanted_word in enumerate(self.wanted_words):
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
            silence_size = int(math.ceil(set_size * self.silence_percentage / 100))
            for _ in range(silence_size):
                self.data_set[set_index].append({
                    'label': SILENCE_LABEL,
                    'file': silence_wav_path,
                    'speaker': "None"
                })

            # Pick some unknowns to add to each partition of the data set.
            random.shuffle(unknown_set[set_index])
            unknown_size = int(math.ceil(set_size * self.unknown_percentage / 100))
            self.data_set[set_index].extend(unknown_set[set_index][:unknown_size])

        # Make sure the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_set[set_index])

        # Prepare the rest of the result data structure.
        self.words_list = prepare_words_list(self.wanted_words)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
            else:
                self.word_to_index[word] = UNKNOWN_WORD_INDEX
        self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

    def generate_background_noise(self):
        """Loads background noise, used to augment clean speech."""

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
        """Returns data set size, given which partition to use, must be 'training', 'validation', or 'testing'."""
        return len(self.data_set[mode])

    def get_data(self, mode):
        """ Retrieves sample data for given self.feature_extraction_method, and performs data augmentation.
        Args:
            mode: Which partition to use, must be 'training', 'validation', or 'testing'.
        Returns:
            List of sample data for the samples, and list of labels in one-hot form.
        """

        if mode != 'training':
            background_frequency = 0
            background_volume = 0
            time_shift_samples = 0
        else:
            background_frequency = self.background_frequency
            background_volume = self.background_volume
            time_shift_samples = self.time_shift_samples

        # Pick one of the partitions to choose samples from
        candidates = self.data_set[mode]
        use_background = (self.background_noise and (mode == 'training'))
        pick_deterministically = (mode != 'training')
        # Pick which audio sample to use.
        if self.batch_size == -1 or pick_deterministically:
            # The randomness is eliminated here to train on the same batch ordering
            # sample_index = i
            sample_index = np.random.randint(len(candidates))
        else:
            sample_index = np.random.randint(len(candidates))
        sample = candidates[sample_index]

        # Compute time shift offset
        if time_shift_samples > 0:
            time_shift_amount = np.random.randint(-time_shift_samples, time_shift_samples)
        else:
            time_shift_amount = 0
        if time_shift_amount > 0:
            time_shift_padding = [[time_shift_amount, 0], [0, 0]]
            time_shift_offset = [0, 0]
        else:
            time_shift_padding = [[0, -time_shift_amount], [0, 0]]
            time_shift_offset = [-time_shift_amount, 0]

        wav_filename = sample['file']

        # Select background noise to mix in.
        if use_background or sample['label'] == SILENCE_LABEL:
            background_index = np.random.randint(len(self.background_noise))
            background_samples = self.background_noise[background_index].numpy()
            assert (len(background_samples) > self.desired_samples)

            background_offset = np.random.randint(0, len(background_samples) - self.desired_samples)
            background_clipped = background_samples[background_offset:(background_offset + self.desired_samples)]
            background_reshaped = background_clipped.reshape([self.desired_samples, 1])

            if sample['label'] == SILENCE_LABEL:
                background_volume = np.random.uniform(0, 1)
            elif np.random.uniform(0, 1) < background_frequency:
                background_volume = np.random.uniform(0, background_volume)
            else:
                background_volume = 0
        else:
            background_reshaped = np.zeros([self.desired_samples, 1])
            background_volume = 0

        background_noise = background_reshaped
        # data_augmentation_parameters['background_volume'] = background_volume

        # For silence samples, remove any sound
        if sample['label'] == SILENCE_LABEL:
            foreground_volume = 0
        else:
            foreground_volume = 1

        # Load data
        try:
            sf_loader, _ = sf.read(wav_filename)
            wav_file = torch.Tensor(np.array([sf_loader]))
        except:
            pass

        # Ensure data length is equal to the number of desired samples
        if len(wav_file[0]) < self.desired_samples:
            wav_file = torch.nn.ConstantPad1d(
                (0, self.desired_samples - len(wav_file[0])), 0)(wav_file[0])
        else:
            wav_file = wav_file[0][:self.desired_samples]
        scaled_foreground = torch.mul(wav_file, foreground_volume)

        # Padding wrt the time shift offset
        pad_tuple = tuple(time_shift_padding[0])
        padded_foreground = torch.nn.ConstantPad1d(pad_tuple, 0)(scaled_foreground)
        sliced_foreground = padded_foreground[time_shift_offset[0]:time_shift_offset[0] + self.desired_samples]

        # Mix in background noise
        background_mul = torch.mul(torch.Tensor(background_noise[:, 0]), background_volume)

        data = torch.add(background_mul, sliced_foreground)  # Size([16000])

        label_index = self.word_to_index[sample['label']]

        return data.to(self.device), label_index

    def get_mfcc(self, sample, mfcc_transformation, spectrogram_length):
        """ Apply MFCC feature extraction to sample.
        Args:
            sample (Tensor): Sample on which to apply the MFCC transformation.
        Returns:
            data (Tensor) : MFCC computation from the sample, using melkwargs parameters.
        """
        # Compute MFCCs - PyTorch
        data = mfcc_transformation(sample)  # shape (feature_bin_count, 51)

        # Cut shape to (feature_bin_count, spectrogram_length)
        data = data[:, :spectrogram_length].transpose(0, 1)
        return data.to(self.device)

    def get_linear_stft(self, sample):
        """ Apply Linear STFT feature extraction to sample.
        Args:
            sample: Sample on which to apply the STFT transformation.
        Returns:
            STFT computation from the sample, using stftkwargs parameters.
        """

        # Compute STFT - librosa
        n_fft = 1024

        data = librosa.stft(y=sample,
                            n_fft=n_fft,
                            hop_length=self.data_processing_parameters['window_stride_samples'],
                            win_length=self.data_processing_parameters['window_size_samples'])  # shape (513, 51)

        # Cut shape to (feature_bin_count, spectrogram_length) and transpose
        data = data[:, :self.data_processing_parameters['spectrogram_length']].transpose()

        # Shift data in [0, 255] interval to match Dory request for uint8 inputs
        data = np.clip(data + 128, 0, 255)
        return data

    def get_melscale_stft(self, sample):
        """ Apply Linear STFT feature extraction to sample.
        Args:
            sample: Sample on which to apply the STFT transformation.
        Returns:
            STFT computation from the sample, using stftkwargs parameters.
        """

        # Compute STFT - librosa
        n_fft = 1024

        data = librosa.stft(y=sample,
                            n_fft=n_fft,
                            hop_length=self.data_processing_parameters['window_stride_samples'],
                            win_length=self.data_processing_parameters['window_size_samples'])  # shape (513, 51)

        data = np.abs(data)
        data = librosa.amplitude_to_db(data, ref=np.max)

        # Cut shape to (feature_bin_count, spectrogram_length) and transpose
        data = data[:, :self.data_processing_parameters['spectrogram_length']].transpose()
        # Shift data in [0, 255] interval to match Dory request for uint8 inputs
        data = np.clip(data + 128, 0, 255)
        return data

    def get_mel_spectrogram(self, sample):
        """ Get the MelSpectrogram from the sample.
        Args:
            sample: Sample on which to apply the transformation.
        Returns:
            MelSpectrogram from the sample, using parameters from data_processing_parameters.
        """

        # Create a data placeholder
        sample = torch.Tensor(sample)
        # Compute MelSpectrogram - PyTorch
        melspect_transformation = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.data_processing_parameters['sample_rate'],
            n_fft=1024, win_length=self.data_processing_parameters['window_size_samples'],
            hop_length=self.data_processing_parameters['window_stride_samples'], f_min=20, f_max=4000, n_mels=10
        )
        data = melspect_transformation(sample)

        # Cut shape to (feature_bin_count, spectrogram_length)
        data = data[:, :self.data_processing_parameters['spectrogram_length']].numpy().transpose()

        # Shift data in [0, 255] interval to match Dory request for uint8 inputs
        data = np.clip(data + 128, 0, 255)
        return data

    def get_lpcc(self, sample, order):
        """ Get the LPCC coefficients from an LPC input.
        Args:
            sample: LPC on which to apply the transformation.
        Returns:
            LPCC derived from the LPC sample.
        """
        nin, ncol = sample.shape
        nout = order + 1

        cep = np.zeros((nout, ncol))
        cep[0, :] = -np.log(sample[0, :])

        norm_a = np.divide(sample, np.add(np.tile(sample[0, :], (nin, 1)), 1e-8))

        for n in range(1, nout):
            sum_var = 0
            for m in range(1, n):
                sum_var = np.add(
                    sum_var,
                    np.multiply(np.multiply((n - m), norm_a[m, :]),
                                cep[(n - m), :]))

            cep[n, :] = -np.add(norm_a[n, :], np.divide(sum_var, n))
        return cep

    def get_lsf(self, sample):  # TODO implement
        """ Get the Line Spectral Frequencies (also Line Spectrum Pairs) from an LPC input.
        Args:
            sample: LPC on which to apply the transformation.
        Returns:
            LSF derived from the LPC sample.
        """
        # the rhs does not have a constant expression and we reverse the coefficients
        print(sample.shape)

        rhs = [0] + sample[::-1] + [1]
        rectify = True
        # The P polynomial
        P = []
        # The Q polynomial
        Q = []
        # Assuming constant coefficient is 1, which is required. Moreover z^{-p+1} does not exist on the lhs, thus appending 0
        sample = [1] + sample[:] + [0]
        for l, r in itertools.zip_longest(sample, rhs):
            P.append(l + r)
            Q.append(l - r)
        # Find the roots of the polynomials P,Q (numpy assumes we have the form of: p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]
        # mso we need to reverse the order)
        print(P[0].shape)
        p_roots = np.roots(P[::-1])
        q_roots = np.roots(Q[::-1])
        # Keep the roots in order
        lsf_p = sorted(np.angle(p_roots))
        lsf_q = sorted(np.angle(q_roots))
        # print sorted(lsf_p+lsf_q),len([i for  i in lsf_p+lsf_q if i > 0.])
        if rectify:
            # We only return the positive elements, and also remove the final Pi (3.14) value at the end,
            # since it always occurs
            return sorted(i for i in lsf_q + lsf_p if (i > 0))[:-1]
        else:
            # Remove the -Pi and +pi at the beginning and end in the list
            return sorted(i for i in lsf_q + lsf_p)[1:-1]


class AudioGenerator(torch.utils.data.Dataset):
    """Returns batches of preprocessed data and labels"""

    def __init__(self, mode, audio_processor):
        self.mode = mode
        self.audio_processor = audio_processor

    def __len__(self):
        """Returns data set length"""
        if self.audio_processor.batch_size == -1:
            return len(self.audio_processor.data_set[self.mode])
        else:
            return int(len(self.audio_processor.data_set[self.mode]) / self.audio_processor.batch_size)

    def __getitem__(self, idx):
        """Returns a random batch of data, unless training_parameters['batch_size'] == -1."""
        data, labels = self.audio_processor.get_data(self.mode)
        return data, labels
