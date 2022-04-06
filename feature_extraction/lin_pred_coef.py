from feature_extraction.dataset import AudioProcessor
from librosa import lpc
import numpy as np
import soundfile as sf
import torch

MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M
BACKGROUND_NOISE_LABEL = '_background_noise_'
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
RANDOM_SEED = 59185


class LPCProcessor(AudioProcessor):  # Uses the lpctorch implementation so far, as it is not proposed by torchaudio
    # Prepare data

    def __init__(self, training_parameters, data_processing_parameters):
        super().__init__(training_parameters, data_processing_parameters)

    def get_data(self, mode, training_parameters):
        # Initial data augmentation phase, taken from mfcc, will see if we refactor in the future
        # Prepare and return data (utterances and labels) for inference

        # Pick one of the partitions to choose samples from
        candidates = self.data_set[mode]
        if training_parameters['batch_size'] == -1:
            samples_number = len(candidates)
        else:
            samples_number = max(0, min(training_parameters['batch_size'], len(candidates)))

        # Create a data placeholder
        data_placeholder = np.zeros((samples_number, self.data_processing_parameters['spectrogram_length'],
                                     self.data_processing_parameters['feature_bin_count']), dtype='float32')
        labels_placeholder = np.zeros(samples_number)

        # Required for noise analysis
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
            background_add = torch.add(background_mul, sliced_foreground)

            # Initialize LPCCoefficients module given params - librosa implementation

            # Get the coefficients given a signal
            # torch.Tensor of size ( Batch, Samples )
            data = lpc(background_add.detach().numpy(), order=32)

            # TODO: see if we want to manage this
            data_placeholder[i] = data[:, :self.data_processing_parameters['spectrogram_length']].numpy().transpose()

            # Shift data in [0, 255] interval to match Dory request for uint8 inputs
            data_placeholder[i] = np.clip(data_placeholder[i] + 128, 0, 255)

            label_index = self.word_to_index[sample['label']]
            labels_placeholder[i] = label_index

        return data_placeholder, labels_placeholder
