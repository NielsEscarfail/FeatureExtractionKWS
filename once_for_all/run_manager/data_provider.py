import random
import time
import warnings
import os
import numpy as np
import torch.utils.data
from torchaudio.transforms import MFCC

from .dataset import AudioGenerator, AudioProcessor


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
            ft_extr_type=["mfcc"],  # "mfcc", "mel_spectrogram", "dwt", "melscale_stft" // "linear_stft" , "lpc", "lpcc"
            ft_extr_params_list=None,
            rank=None,
            n_worker=0
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

        train_loader_class = torch.utils.data.DataLoader
        train_dataset = self.train_dataset()

        print("Feature extraction type : ", ft_extr_type)
        print("Parameters used : ", ft_extr_params_list)
        print("augmented audio shape: ", train_dataset.__getitem__(0)[0].shape)
        print("Valid_size : ", valid_size)
        print("Train batch size : ", train_batch_size)

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
                pin_memory=False,
                collate_fn=self.collate_batch
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=test_batch_size,
                sampler=valid_sampler,
                num_workers=n_worker,
                pin_memory=False,
                collate_fn=self.collate_batch
            )
        else:
            print("train loader class after")
            self.train = train_loader_class(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=n_worker,
                pin_memory=False,
                collate_fn=self.collate_batch
            )
            self.valid = None

        test_dataset = self.test_dataset()

        self.test = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=True,
            num_workers=n_worker,
            pin_memory=False,
            collate_fn=self.collate_batch
        )

        valid_dataset = self.valid_dataset()
        self.valid = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=test_batch_size,
            shuffle=True,
            num_workers=n_worker,
            pin_memory=False,
            collate_fn=self.collate_batch
        )

        if self.valid is None:
            self.valid = self.test

        print("Train length: ", len(self.train))

    def collate_batch_subtrain(self, batch):
        """Collates batches deterministically based on self.ft_extr_type and self.active_ft_extr_params."""
        data_placeholder = []
        labels_placeholder = []

        transformation = self.active_ft_extr_type
        if transformation == 'mfcc':
            feature_bin_count = self.active_ft_extr_params[0]
            spectrogram_length = self.active_ft_extr_params[1]

        for (data, label) in batch:
            # Apply transformation
            if transformation == 'mfcc':
                # data = self.audio_processor.get_mfcc(data, feature_bin_count, spectrogram_length)[None, :, :]
                data = torch.unsqueeze(self.audio_processor.get_mfcc(data, feature_bin_count, spectrogram_length),
                                       dim=0)
            else:
                raise NotImplementedError

            data_placeholder.append(data)
            labels_placeholder.append(label)

        return torch.stack(data_placeholder, dim=0).to(self.device), torch.tensor(labels_placeholder).to(self.device)

    def collate_batch(self, batch):
        """Collates batches and applies self.ft_extr_type.
         Randomly picking parameters from self.ft_extr_params_list for each batch."""

        start = time.time()
        data_placeholder = []
        labels_placeholder = []

        transformation = self.ft_extr_type
        ft_extr_params = random.choice(self.ft_extr_params_list)
        if transformation == 'mfcc':
            feature_bin_count = ft_extr_params[0]
            spectrogram_length = ft_extr_params[1]

            melkwargs = {'n_fft': 1024, 'win_length': self.audio_processor.window_size_samples,
                         'hop_length': self.audio_processor.window_stride_samples,
                         'f_min': 20, 'f_max': 4000, 'n_mels': 40}

            mfcc_transformation = MFCC(
                n_mfcc=feature_bin_count,
                sample_rate=self.audio_processor.desired_samples, melkwargs=melkwargs, log_mels=True,
                norm='ortho').to(self.device)

        for (data, label) in batch:
            # Apply transformation
            if transformation == 'mfcc':
                data = self.audio_processor.get_mfcc(data, mfcc_transformation, spectrogram_length)
                data = torch.unsqueeze(data, dim=0)
            else:
                raise NotImplementedError

            data_placeholder.append(data)
            labels_placeholder.append(label)

        end = time.time()
        print("FT batch : ", end - start)

        return torch.stack(data_placeholder, dim=0), torch.tensor(labels_placeholder)

    @staticmethod
    def name():
        return "speech-commands"

    @property
    def data_shape(self):  # TODO n_channels
        return 1, self.active_ft_extr_params[0], self.active_ft_extr_params[1]  # C, H, W

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
        # self.active_ft_extr_type = new_ft_type

    def collate_batch_subtrain(self, batch):
        data_placeholder = []
        labels_placeholder = []

        transformation = self.active_ft_extr_type
        if transformation == 'mfcc':
            feature_bin_count = self.active_ft_extr_params[0]
            spectrogram_length = self.active_ft_extr_params[1]

        for (data, label) in batch:
            # Apply transformation
            if transformation == 'mfcc':
                data = torch.unsqueeze(self.audio_processor.get_mfcc(data, feature_bin_count, spectrogram_length),
                                       dim=0)
                # data = self.audio_processor.get_mfcc(data, feature_bin_count, spectrogram_length)[None, :, :]
            else:
                raise NotImplementedError

            # Create feature extraction batch
            data_placeholder.append(data)
            labels_placeholder.append(label)

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
                pin_memory=False,
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
