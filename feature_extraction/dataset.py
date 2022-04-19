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
import librosa
import pywt

# from kaldi.feat import PLP

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

    def __init__(self, training_parameters, data_processing_parameters):
        self.data_directory = training_parameters['data_dir']
        self.data_processing_parameters = data_processing_parameters
        self.feature_extraction_method = data_processing_parameters['feature_extraction_method']

        self.generate_background_noise()
        self.generate_data_dictionary(training_parameters)

    def generate_data_dictionary(self, training_parameters):
        """ For each data set, generate a dictionary containing the path to each file, its label, and its speaker.
        Args:
            training_parameters: data and model parameters, described at config.yaml.
        """

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

    def get_raw_data(self, mode, training_parameters):
        """ Retrieve sample data with no transformation. Will be set as get_data if feature_extraction_method == 'raw'.
        Args:
          mode: Which partition to use, must be 'training', 'validation', or 'testing'.
          training_parameters: data and model parameters, described at config.yaml
        Returns:
          List of sample data for the samples, and list of labels in one-hot form.
        """
        # Pick one of the partitions to choose samples from
        candidates = self.data_set[mode]
        if training_parameters['batch_size'] == -1:
            samples_number = len(candidates)
        else:
            samples_number = max(0, min(training_parameters['batch_size'], len(candidates)))

        desired_samples = self.data_processing_parameters['desired_samples']

        # Create a data placeholder
        data_placeholder = np.zeros((samples_number, desired_samples))
        labels_placeholder = np.zeros(samples_number)

        pick_deterministically = (mode != 'training')

        for i in range(0, samples_number):
            # Pick which audio sample to use.
            if training_parameters['batch_size'] == -1 or pick_deterministically:
                # The randomness is eliminated here to train on the same batch ordering
                sample_index = i
            else:
                sample_index = np.random.randint(len(candidates))
            sample = candidates[sample_index]

            data_augmentation_parameters = {
                'wav_filename': sample['file'],
            }

            # For silence samples, remove any sound
            if sample['label'] == SILENCE_LABEL:
                data_augmentation_parameters['foreground_volume'] = 0
            else:
                data_augmentation_parameters['foreground_volume'] = 1

            # Load data
            try:
                sf_loader, _ = sf.read(data_augmentation_parameters['wav_filename'])
                wav_file = torch.Tensor(np.array([sf_loader]))
            except:
                pass

            # Ensure data length is equal to the number of desired samples
            if len(wav_file[0]) < desired_samples:
                data = torch.nn.ConstantPad1d(
                    (0, desired_samples - len(wav_file[0])), 0)(wav_file[0])
            else:
                data = wav_file[0][:desired_samples]

            data_placeholder[i] = data

            label_index = self.word_to_index[sample['label']]
            labels_placeholder[i] = label_index
        return data_placeholder, labels_placeholder

    def get_augmented_data(self, mode, training_parameters):
        """ Retrieve sample data and perform data augmentation (shifting, scaling, background noise).
        Args:
            mode: Which partition to use, must be 'training', 'validation', or 'testing'.
            In training mode, data augmentation is performed.
            training_parameters: data parameters, described at config.yaml
        Returns:
            List of sample data for the samples, and list of labels in one-hot form.
        """

        # Pick one of the partitions to choose samples from
        candidates = self.data_set[mode]
        if training_parameters['batch_size'] == -1:
            samples_number = len(candidates)
        else:
            samples_number = max(0, min(training_parameters['batch_size'], len(candidates)))

        desired_samples = self.data_processing_parameters['desired_samples']

        # Create a data placeholder
        data_placeholder = np.zeros((samples_number, desired_samples))
        labels_placeholder = np.zeros(samples_number)

        use_background = (self.background_noise and (mode == 'training'))
        pick_deterministically = (mode != 'training')

        for i in range(0, samples_number):

            # Pick which audio sample to use.
            if training_parameters['batch_size'] == -1 or pick_deterministically:
                # The randomness is eliminated here to train on the same batch ordering
                sample_index = i
            else:
                sample_index = np.random.randint(len(candidates))
            sample = candidates[sample_index]

            # Compute time shift offset
            if training_parameters['time_shift_samples'] > 0:
                time_shift_amount = np.random.randint(-training_parameters['time_shift_samples'],
                                                      training_parameters['time_shift_samples'])
            else:
                time_shift_amount = 0
            if time_shift_amount > 0:
                time_shift_padding = [[time_shift_amount, 0], [0, 0]]
                time_shift_offset = [0, 0]
            else:
                time_shift_padding = [[0, -time_shift_amount], [0, 0]]
                time_shift_offset = [-time_shift_amount, 0]

            data_augmentation_parameters = {
                'wav_filename': sample['file'],
                'time_shift_padding': time_shift_padding,
                'time_shift_offset': time_shift_offset,
            }

            # Select background noise to mix in.
            if use_background or sample['label'] == SILENCE_LABEL:
                background_index = np.random.randint(len(self.background_noise))
                background_samples = self.background_noise[background_index].numpy()
                assert (len(background_samples) > self.data_processing_parameters['desired_samples'])

                background_offset = np.random.randint(0, len(background_samples) - self.data_processing_parameters[
                    'desired_samples'])
                background_clipped = background_samples[background_offset:(
                        background_offset + self.data_processing_parameters['desired_samples'])]
                background_reshaped = background_clipped.reshape(
                    [self.data_processing_parameters['desired_samples'], 1])

                if sample['label'] == SILENCE_LABEL:
                    background_volume = np.random.uniform(0, 1)
                elif np.random.uniform(0, 1) < training_parameters['background_frequency']:
                    background_volume = np.random.uniform(0, training_parameters['background_volume'])
                else:
                    background_volume = 0
            else:
                background_reshaped = np.zeros([self.data_processing_parameters['desired_samples'], 1])
                background_volume = 0

            data_augmentation_parameters['background_noise'] = background_reshaped
            data_augmentation_parameters['background_volume'] = background_volume

            # For silence samples, remove any sound
            if sample['label'] == SILENCE_LABEL:
                data_augmentation_parameters['foreground_volume'] = 0
            else:
                data_augmentation_parameters['foreground_volume'] = 1

            # Load data
            try:
                sf_loader, _ = sf.read(data_augmentation_parameters['wav_filename'])
                wav_file = torch.Tensor(np.array([sf_loader]))
            except:
                pass

            # Ensure data length is equal to the number of desired samples
            if len(wav_file[0]) < self.data_processing_parameters['desired_samples']:
                wav_file = torch.nn.ConstantPad1d(
                    (0, self.data_processing_parameters['desired_samples'] - len(wav_file[0])), 0)(wav_file[0])
            else:
                wav_file = wav_file[0][:self.data_processing_parameters['desired_samples']]
            scaled_foreground = torch.mul(wav_file, data_augmentation_parameters['foreground_volume'])

            # Padding wrt the time shift offset
            pad_tuple = tuple(data_augmentation_parameters['time_shift_padding'][0])
            padded_foreground = torch.nn.ConstantPad1d(pad_tuple, 0)(scaled_foreground)
            sliced_foreground = padded_foreground[data_augmentation_parameters['time_shift_offset'][0]:
                                                  data_augmentation_parameters['time_shift_offset'][0] +
                                                  self.data_processing_parameters['desired_samples']]

            # Mix in background noise
            background_mul = torch.mul(torch.Tensor(data_augmentation_parameters['background_noise'][:, 0]),
                                       data_augmentation_parameters['background_volume'])

            data = torch.add(background_mul, sliced_foreground)  # Size([16000])

            data_placeholder[i] = data.numpy().transpose()

            # Shift data in [0, 255] interval to match Dory request for uint8 inputs
            # data_placeholder[i] = np.clip(data_placeholder[i] + 128, 0, 255)

            label_index = self.word_to_index[sample['label']]
            labels_placeholder[i] = label_index

        return data_placeholder, labels_placeholder

    def get_mfcc(self, sample):
        """ Apply MFCC feature extraction to sample.
        Args:
            sample: Sample on which to apply the MFCC transformation.
        Returns:
            MFCC computation from the sample, using melkwargs parameters.
        """

        # Create a data placeholder
        sample = torch.Tensor(sample)
        # Compute MFCCs - PyTorch
        melkwargs = {'n_fft': 1024, 'win_length': self.data_processing_parameters['window_size_samples'],
                     'hop_length': self.data_processing_parameters['window_stride_samples'],
                     'f_min': 20, 'f_max': 4000, 'n_mels': 40}
        mfcc_transformation = torchaudio.transforms.MFCC(
            n_mfcc=self.data_processing_parameters['feature_bin_count'],
            sample_rate=self.data_processing_parameters['desired_samples'], melkwargs=melkwargs, log_mels=True,
            norm='ortho')
        data = mfcc_transformation(sample)  # shape (feature_bin_count, 51)

        # Cut shape to (feature_bin_count, spectrogram_length)
        data = data[:, :self.data_processing_parameters['spectrogram_length']].numpy().transpose()

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

    def get_data(self, mode, training_parameters):
        """ Retrieve sample data for given self.feature_extraction_method.
        Args:
            mode: Which partition to use, must be 'training', 'validation', or 'testing'.
            training_parameters: data parameters, described at config.yaml
        Returns:
            List of sample data for the samples, and list of labels in one-hot form.
        """
        if self.feature_extraction_method == 'raw':
            return self.get_raw_data(mode, training_parameters)

        elif self.feature_extraction_method == 'augmented':
            return self.get_augmented_data(mode, training_parameters)

        elif self.feature_extraction_method == 'mfcc':  # for now, always uses augmented data
            data, labels = self.get_augmented_data(mode, training_parameters)
            data = np.apply_along_axis(self.get_mfcc, 1, data)
            return data, labels

        elif self.feature_extraction_method == 'mel_spectrogram':
            data, labels = self.get_augmented_data(mode, training_parameters)
            data = np.apply_along_axis(self.get_mel_spectrogram, 1, data)
            return data, labels

        elif self.feature_extraction_method == 'lpc':
            order = 30
            data, labels = self.get_augmented_data(mode, training_parameters)
            data = np.apply_along_axis(librosa.lpc, 1, data, order=order)
            return data, labels

        elif self.feature_extraction_method == 'lpcc':
            order = 30
            # get LPC coefficients
            data, labels = self.get_augmented_data(mode, training_parameters)
            data = np.apply_along_axis(librosa.lpc, 1, data, order=order)
            # Derive LPCC coefficients
            data = self.get_lpcc(data, order)
            return data, labels

        elif self.feature_extraction_method == 'lsf':  # TODO
            order = 30
            # get LPC coefficients
            data, labels = self.get_augmented_data(mode, training_parameters)
            data = np.apply_along_axis(librosa.lpc, 1, data, order=order)

            # Derive LSF coefficients
            data = self.get_lsf(data)
            return data, labels

        elif self.feature_extraction_method == 'dwt':
            # data, labels = self.get_augmented_data(mode, training_parameters)
            data, labels = self.get_raw_data(mode, training_parameters)
            (cA, cD) = pywt.dwt(data, 'db1')  # Approximation and detail coefficients
            """'haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus',
            'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor'"""
            # cf https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
            # print(pywt.wavelist())
            data = np.concatenate((cA, cD), axis=-1)  # verify if we want to keep it (batch_size, 16000)
            # The data contains all information from the original set (signal can be recovered),
            # it would be interesting to add an extra step from here before running the model.
            # Might also explore normalizing / treating sides
            return data, labels

        elif self.feature_extraction_method == 'plp':  # TODO
            data, labels = self.get_augmented_data(mode, training_parameters)
            data = PLP(data)
            return data, labels

        else:
            return NotImplementedError("Feature extraction method is not implemented.")


class AudioGenerator(torch.utils.data.Dataset):
    """Returns batches of preprocessed data and labels"""

    def __init__(self, mode, audio_processor, training_parameters):
        self.mode = mode
        self.audio_processor = audio_processor
        if self.mode != 'training':
            training_parameters['background_frequency'] = 0
            training_parameters['background_volume'] = 0
            training_parameters['time_shift_samples'] = 0
        self.training_parameters = training_parameters

    def __len__(self):
        """Returns data set length"""
        if self.training_parameters['batch_size'] == -1:
            return len(self.audio_processor.data_set[self.mode])
        else:
            return int(len(self.audio_processor.data_set[self.mode]) / self.training_parameters['batch_size'])

    def __getitem__(self, idx):
        """Returns a random batch of data, unless training_parameters['batch_size'] == -1."""
        data, labels = self.audio_processor.get_data(self.mode, self.training_parameters)
        return data, labels
