import random
import time
import warnings
import os
import numpy as np
import torch.utils.data
import torchaudio
from torchaudio.transforms import MFCC
import librosa
from spafe.features.rplp import plp
from spafe.features.ngcc import ngcc

from .dataset import AudioGenerator, AudioProcessor


# torch.multiprocessing.set_start_method('spawn')  # good solution !!!!


class KWSDataProvider:
    """ Handles data providing and feature extraction. """
    DEFAULT_PATH = "/dataset/speech_commands_v0.02"
    SUB_SEED = 937162211  # random seed for sampling subset
    VALID_SEED = 2147483647  # random seed for the validation set

    def __init__(
            self,
            save_path=None,
            train_batch_size=256,
            test_batch_size=512,
            valid_size=None,
            ft_extr_type=["mel_spectrogram"],
            # "mfcc", "mel_spectrogram", "dwt", "melscale_stft" // "linear_stft" , "lpc", "lpcc"
            ft_extr_params_list=None,
            rank=None,
            n_worker=4
    ):

        warnings.filterwarnings("ignore")
        self._save_path = save_path
        # move network to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.audio_processor = AudioProcessor(feature_extraction_method=ft_extr_type)

        self.ft_extr_type = ft_extr_type
        self.ft_extr_params_list = ft_extr_params_list

        self.active_ft_extr_type = self.ft_extr_type
        self.active_ft_extr_params = self.ft_extr_params_list[0]
        self.active_transformation = None

        self.init_transformations()

        train_loader_class = torch.utils.data.DataLoader
        train_dataset = self.train_dataset()

        print("Feature extraction type : ", ft_extr_type)
        print("Parameters used : ", ft_extr_params_list)

        if valid_size is not None:
            if not isinstance(valid_size, int):
                assert isinstance(valid_size, float) and 0 < valid_size < 1
                valid_size = int(len(train_dataset) * valid_size)

            valid_dataset = self.valid_dataset()
            train_indexes, valid_indexes = self.random_sample_valid_set(
                len(train_dataset), valid_size
            )

            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                train_indexes
            )
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                valid_indexes
            )

            self.train = train_loader_class(
                train_dataset,
                batch_size=train_batch_size,
                sampler=train_sampler,
                num_workers=n_worker,
                pin_memory=True,
                collate_fn=self.collate_batch
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=test_batch_size,
                sampler=valid_sampler,
                num_workers=n_worker,
                pin_memory=True,
                collate_fn=self.collate_batch_subtrain
            )
        else:  # No validation set
            self.train = train_loader_class(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=n_worker,
                pin_memory=True,
                collate_fn=self.collate_batch
            )
            self.valid = None

        # Set up test dataset
        test_dataset = self.test_dataset()

        self.test = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=True,
            num_workers=n_worker,
            pin_memory=True,
            collate_fn=self.collate_batch_subtrain
        )

        if self.valid is None:
            self.valid = self.test

        print("Train length: ", len(self.train))
        print("Valid length: ", len(self.valid))
        print("Test length: ", len(self.test))

    @staticmethod
    def name():
        return "speech-commands"

    @property
    def data_shape(self):
        if self.ft_extr_type == "lpcc" or self.ft_extr_type == "plp":
            return 1, 1, self.active_ft_extr_params + 1
        else:
            return 1, self.active_ft_extr_params[0], self.active_ft_extr_params[1]

    @property
    def n_classes(self):
        return 12

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser("~/dataset/speech_commands_v0.02")
        return self._save_path

    @property
    def data_url(self):
        raise ValueError("unable to download %s" % self.name())

    def train_dataset(self):
        train_dataset = AudioGenerator(mode='training', audio_processor=self.audio_processor)
        return train_dataset

    def test_dataset(self):
        test_dataset = AudioGenerator(mode='testing', audio_processor=self.audio_processor)
        return test_dataset

    def valid_dataset(self):
        valid_dataset = AudioGenerator(mode='validation', audio_processor=self.audio_processor)
        return valid_dataset

    @property
    def train_path(self):
        return os.path.join(self.save_path, "train")

    @property
    def valid_path(self):
        return os.path.join(self.save_path, "val")

    def assign_active_ft_extr_params(self, new_ft_extr_params):
        self.active_ft_extr_params = new_ft_extr_params
        if self.ft_extr_type == 'mfcc' or self.ft_extr_type == 'mel_spectrogram':
            transformation_idx = self.ft_extr_params_list.index(new_ft_extr_params)
            self.active_transformation = self.transformations[transformation_idx]

    def init_transformations(self):
        """ Transformations:
        mfcc:
            n_mfcc : number of mel bins
            win_size_ms : window size, in ms ; window_stride_ms is set to window_size/2

        mel_spectrogram
            n_mels : number of mel bins
            win_size_ms : window size, in ms ; window_stride_ms is set to window_size/2

        linear_stft
            n_fft : number of ffts
            win_size_ms : window size, in ms ; window_stride_ms is set to window_size/2
        """
        # TODO check n_mels + window_size_samples + window_stride_samples
        self.transformations = []
        if self.ft_extr_type == 'mfcc':
            for (n_mfcc, win_size_ms) in self.ft_extr_params_list:  # resolution instead of crop
                window_stride_ms = win_size_ms / 2
                win_length = int(self.audio_processor.desired_samples * win_size_ms / 1000)
                hop_length = int(self.audio_processor.desired_samples * window_stride_ms / 1000)
                melkwargs = {'n_fft': 2048, 'win_length': win_length,
                             'hop_length': hop_length,
                             'f_min': 20, 'f_max': 4000, 'n_mels': n_mfcc}  # this is resolution
                mfcc_transformation = MFCC(
                    n_mfcc=16,  # this is crop
                    sample_rate=self.audio_processor.desired_samples, melkwargs=melkwargs, log_mels=True,
                    norm='ortho')
                self.transformations.append(mfcc_transformation)

        elif self.ft_extr_type == 'mel_spectrogram':
            for (n_mels, win_size_ms) in self.ft_extr_params_list:
                window_stride_ms = win_size_ms / 2
                win_length = int(self.audio_processor.desired_samples * win_size_ms / 1000)
                hop_length = int(self.audio_processor.desired_samples * window_stride_ms / 1000)

                melspect_transformation = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.audio_processor.desired_samples,
                    n_fft=2048, win_length=win_length,
                    hop_length=hop_length, f_min=20, f_max=4000, n_mels=n_mels
                )
                self.transformations.append(melspect_transformation)

    def collate_batch(self, batch):
        """Collates batches and applies self.ft_extr_type.
         Randomly picking parameters from self.ft_extr_params_list for each batch."""

        data_placeholder = []
        labels_placeholder = []

        transformation_type = self.ft_extr_type
        ft_extr_params = random.choice(self.ft_extr_params_list)

        # No transformations
        if transformation_type == 'raw':
            for (data, label) in batch:
                data = data.reshape((125, 128))
                data = torch.unsqueeze(data, dim=0)
                data_placeholder.append(data)
                labels_placeholder.append(label)

        # Preloaded transformations
        elif transformation_type == 'mfcc' or transformation_type == 'mel_spectrogram':
            transformation_idx = self.ft_extr_params_list.index(ft_extr_params)
            transformation = self.transformations[transformation_idx]
            for (data, label) in batch:
                data = transformation(data)
                data = torch.unsqueeze(data, dim=0)
                data_placeholder.append(data)
                labels_placeholder.append(label)

        # Transformation functions
        elif transformation_type == 'linear_stft':
            n_fft, win_size_ms = ft_extr_params
            window_stride_ms = win_size_ms / 2
            win_length = int(self.audio_processor.desired_samples * win_size_ms / 1000)
            hop_length = int(self.audio_processor.desired_samples * window_stride_ms / 1000)

            for (data, label) in batch:
                data = torch.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
                data_placeholder.append(data)
                labels_placeholder.append(label)

        elif transformation_type == 'lpcc':
            order = ft_extr_params
            for i, (data, label) in enumerate(batch):
                # compute lpcs
                data = librosa.lpc(np.array(data), order=order)
                # compute lpccs
                data = self.audio_processor.get_lpcc(sample=data[:, None], order=order)
                data = torch.tensor(data).float()
                data = torch.unsqueeze(data, dim=0)
                data_placeholder.append(data)
                labels_placeholder.append(label)

        elif transformation_type == 'plp':
            order = ft_extr_params
            for data, label in batch:
                data = plp(sig=data,
                           fs=self.audio_processor.desired_samples,
                           modelorder=order)
                data = torch.tensor(data).float()
                data = torch.unsqueeze(data, dim=0)
                data_placeholder.append(data)
                labels_placeholder.append(label)

        else:
            raise NotImplementedError

        return torch.stack(data_placeholder, dim=0), torch.tensor(labels_placeholder)

    def collate_batch_subtrain(self, batch):
        """Collates batches deterministically based on self.ft_extr_type and self.active_ft_extr_params."""
        data_placeholder = []
        labels_placeholder = []

        transformation_type = self.active_ft_extr_type
        active_transformation = self.active_transformation
        active_ft_extr_params = self.active_ft_extr_params

        # No transformations
        if transformation_type == 'raw':
            for (data, label) in batch:
                data = data.reshape((125, 128))
                data = torch.unsqueeze(data, dim=0)
                data_placeholder.append(data)
                labels_placeholder.append(label)

        # Preloaded transformations
        elif transformation_type == 'mfcc' or transformation_type == 'mel_spectrogram':
            for (data, label) in batch:
                data = active_transformation(data)
                data = torch.unsqueeze(data, dim=0)
                data_placeholder.append(data)
                labels_placeholder.append(label)

        # Transformation functions
        elif transformation_type == 'linear_stft':
            n_fft, win_size_ms = active_ft_extr_params
            window_stride_ms = win_size_ms / 2
            win_length = int(self.audio_processor.desired_samples * win_size_ms / 1000)
            hop_length = int(self.audio_processor.desired_samples * window_stride_ms / 1000)

            for (data, label) in batch:
                data = torch.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
                data = torch.unsqueeze(data, dim=0)
                data_placeholder.append(data)
                labels_placeholder.append(label)

        elif transformation_type == 'lpcc':
            order = active_ft_extr_params

            for i, (data, label) in enumerate(batch):
                # compute lpcs
                data = librosa.lpc(np.array(data), order=order)

                # compute lpccs
                data = self.audioprocessor.get_lpcc(sample=data[:, None], order=order)
                data = torch.tensor(data).float()
                data = torch.unsqueeze(data, dim=0)
                data_placeholder.append(data)
                labels_placeholder.append(label)

        elif transformation_type == 'plp':
            order = active_ft_extr_params
            for i, (data, label) in enumerate(batch):
                # compute lpcs
                data = plp(sig=data,
                           fs=self.audio_processor.desired_samples,
                           modelorder=order)
                data = torch.tensor(data).float()
                data = torch.unsqueeze(data, dim=0)
                data_placeholder.append(data)
                labels_placeholder.append(label)

        else:
            raise NotImplementedError

        return torch.stack(data_placeholder, dim=0), torch.tensor(labels_placeholder)

    def build_sub_train_loader(
            self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None
    ):
        # used for resetting BN running statistics
        if self.__dict__.get("sub_train_%s" % self.active_ft_extr_type, None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers

            n_samples = len(self.train.dataset)
            g = torch.Generator()
            g.manual_seed(KWSDataProvider.SUB_SEED)
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()

            new_train_dataset = self.train_dataset()
            chosen_indexes = rand_indexes[:n_images]
            if num_replicas is not None:
                sub_sampler = MyDistributedSampler(
                    new_train_dataset,
                    num_replicas,
                    rank,
                    True,
                    np.array(chosen_indexes),
                )
            else:
                sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    chosen_indexes
                )
            sub_data_loader = torch.utils.data.DataLoader(
                new_train_dataset,
                batch_size=batch_size,
                sampler=sub_sampler,
                num_workers=num_worker,
                pin_memory=True,
                collate_fn=self.collate_batch_subtrain
            )
            self.__dict__["sub_train_%s" % self.active_ft_extr_type] = []
            for images, labels in sub_data_loader:
                self.__dict__["sub_train_%s" % self.active_ft_extr_type].append(
                    (images, labels)
                )
        return self.__dict__["sub_train_%s" % self.active_ft_extr_type]

    @staticmethod
    def random_sample_valid_set(train_size, valid_size):
        assert train_size > valid_size

        g = torch.Generator()
        g.manual_seed(
            KWSDataProvider.VALID_SEED
        )  # set random seed before sampling validation set
        rand_indexes = torch.randperm(train_size, generator=g).tolist()

        valid_indexes = rand_indexes[:valid_size]
        train_indexes = rand_indexes[valid_size:]
        return train_indexes, valid_indexes

    @staticmethod
    def labels_to_one_hot(n_classes, labels):
        new_labels = np.zeros((labels.shape[0], n_classes), dtype=np.float32)
        new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
        return new_labels
