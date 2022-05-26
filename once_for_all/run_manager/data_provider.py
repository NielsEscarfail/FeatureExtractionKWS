import warnings
import os
import math
import numpy as np
import torch.utils.data
from .dataset import AudioGenerator, AudioProcessor
from utils_old import parameter_generation


class KWSDataProvider:
    DEFAULT_PATH = "/dataset/speech_commands_v0.02"

    def __init__(
            self,
            save_path=None,
            train_batch_size=256,
            test_batch_size=512,
            valid_size=None,
            ft_extr_size=(49, 10),
            rank=None,
            n_worker=0
    ):

        warnings.filterwarnings("ignore")
        self._save_path = save_path
        self.ft_extr_size = ft_extr_size  # tuple or list of tuples

        self._valid_transform_dict = {}
        if not isinstance(self.ft_extr_size, tuple):  # Multiple feature extraction methods
            from ofa.utils.my_dataloader import MyDataLoader  # TODO

            assert isinstance(self.ft_extr_size, list)
            self.ft_extr_size.sort()  # e.g., 160 -> 224
            # MyRandomResizedCrop.IMAGE_SIZE_LIST = self.ft_extr_size.copy()
            # MyRandomResizedCrop.ACTIVE_SIZE = max(self.ft_extr_size)

            for input_size in self.ft_extr_size:
                self._valid_transform_dict[input_size] = self.build_valid_transform(
                    input_size
                )
            self.active_img_size = max(self.image_size)  # active resolution for test
            valid_transforms = self._valid_transform_dict[self.active_img_size]
            train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
        else:  # Only one feature extraction method used
            self.active_ft_extr_size = self.ft_extr_size
            # valid_transforms = self.build_valid_transform()
            train_loader_class = torch.utils.data.DataLoader

        # Parameter generation
        # training_parameters, data_processing_parameters, model_parameters = parameter_generation("dscnn", "mfcc")

        # audio_processor = AudioProcessor(training_parameters, data_processing_parameters)
        # train_dataset = AudioGenerator(mode='training', audio_processor=audio_processor, training_parameters=training_parameters)
        # train_dataset = self.train_dataset(self.build_train_transform())
        train_dataset = self.train_dataset()
        print("image shape: ", train_dataset.__getitem__(0)[0].shape)
        print("valid_size : ", valid_size)
        # train_batch_size = 64
        print("train batch size : ", train_batch_size)
        if valid_size is not None:
            if not isinstance(valid_size, int):
                assert isinstance(valid_size, float) and 0 < valid_size < 1
                valid_size = int(len(train_dataset) * valid_size)

            valid_dataset = self.train_dataset(valid_transforms)
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
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=test_batch_size,
                sampler=valid_sampler,
                num_workers=n_worker,
                pin_memory=True,
            )
        else:
            self.train = train_loader_class(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=n_worker,
                pin_memory=True,
            )
            self.valid = None

        test_dataset = self.test_dataset()

        self.test = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=True,
            num_workers=n_worker,
            pin_memory=True,
        )

        valid_dataset = self.valid_dataset()
        self.valid = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=test_batch_size,
            shuffle=True,
            num_workers=n_worker,
            pin_memory=True,
        )

        if self.valid is None:
            self.valid = self.test

        print("Train length: ", len(self.train))


    @staticmethod
    def name():
        return "speech-commands"

    @property
    def data_shape(self):  # TODO n_channels
        return 1, self.active_ft_extr_size[0], self.active_ft_extr_size[1]  # C, H, W

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
        training_parameters, data_processing_parameters, model_parameters = parameter_generation("dscnn", "mfcc")
        audio_processor = AudioProcessor(training_parameters, data_processing_parameters)
        train_dataset = AudioGenerator(mode='training', audio_processor=audio_processor,
                                       training_parameters=training_parameters)
        return train_dataset

    def test_dataset(self):
        training_parameters, data_processing_parameters, model_parameters = parameter_generation("dscnn", "mfcc")
        audio_processor = AudioProcessor(training_parameters, data_processing_parameters)
        test_dataset = AudioGenerator(mode='testing', audio_processor=audio_processor,
                                      training_parameters=training_parameters)
        return test_dataset

    def valid_dataset(self):
        training_parameters, data_processing_parameters, model_parameters = parameter_generation("dscnn", "mfcc")
        audio_processor = AudioProcessor(training_parameters, data_processing_parameters)
        valid_dataset = AudioGenerator(mode='validation', audio_processor=audio_processor,
                                      training_parameters=training_parameters)
        return valid_dataset

    @property
    def train_path(self):
        return os.path.join(self.save_path, "train")

    @property
    def valid_path(self):
        return os.path.join(self.save_path, "val")

    """
    @property
    def normalize(self):
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def build_train_transform(self, image_size=None, print_log=True):
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print(
                "Color jitter: %s, resize_scale: %s, img_size: %s"
                % (self.distort_color, self.resize_scale, image_size)
            )

        if isinstance(image_size, list):
            resize_transform_class = MyRandomResizedCrop
            print(
                "Use MyRandomResizedCrop: %s, \t %s"
                % MyRandomResizedCrop.get_candidate_image_size(),
                "sync=%s, continuous=%s"
                % (
                    MyRandomResizedCrop.SYNC_DISTRIBUTED,
                    MyRandomResizedCrop.CONTINUOUS,
                ),
            )
        else:
            resize_transform_class = transforms.RandomResizedCrop

        # random_resize_crop -> random_horizontal_flip
        train_transforms = [
            resize_transform_class(image_size, scale=(self.resize_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]

        # color augmentation (optional)
        color_transform = None
        if self.distort_color == "torch":
            color_transform = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        elif self.distort_color == "tf":
            color_transform = transforms.ColorJitter(
                brightness=32.0 / 255.0, saturation=0.5
            )
        if color_transform is not None:
            train_transforms.append(color_transform)

        train_transforms += [
            transforms.ToTensor(),
            self.normalize,
        ]

        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        return transforms.Compose(
            [
                transforms.Resize(int(math.ceil(image_size / 0.875))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[
                self.active_img_size
            ] = self.build_valid_transform()
        # change the transform of the valid and test set
        self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]

    def build_sub_train_loader(
            self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None
    ):
        # used for resetting BN running statistics
        if self.__dict__.get("sub_train_%d" % self.active_img_size, None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers

            n_samples = len(self.train.dataset)
            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()

            new_train_dataset = self.train_dataset(
                self.build_train_transform(
                    image_size=self.active_img_size, print_log=False
                )
            )
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
            )
            self.__dict__["sub_train_%d" % self.active_img_size] = []
            for images, labels in sub_data_loader:
                self.__dict__["sub_train_%d" % self.active_img_size].append(
                    (images, labels)
                )
        return self.__dict__["sub_train_%d" % self.active_img_size]

    @staticmethod
    def random_sample_valid_set(train_size, valid_size):
        assert train_size > valid_size

        g = torch.Generator()
        g.manual_seed(
            DataProvider.VALID_SEED
        )  # set random seed before sampling validation set
        rand_indexes = torch.randperm(train_size, generator=g).tolist()

        valid_indexes = rand_indexes[:valid_size]
        train_indexes = rand_indexes[valid_size:]
        return train_indexes, valid_indexes
    """
    @staticmethod
    def labels_to_one_hot(n_classes, labels):
        new_labels = np.zeros((labels.shape[0], n_classes), dtype=np.float32)
        new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
        return new_labels
