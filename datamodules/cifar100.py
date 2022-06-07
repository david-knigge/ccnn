import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl

import numpy as np

# config
from hydra import utils


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        test_batch_size,
        data_type,
        num_workers,
        pin_memory,
        **kwargs,
    ):
        super().__init__()

        # Save parameters to self
        self.data_dir = utils.get_original_cwd() + data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Determine data_type
        if data_type == "default":
            self.data_type = "image"
            self.data_dim = 2
        else:
            raise ValueError(f"data_type {data_type} not supported.")

        self.input_channels = 3
        self.output_channels = 100

        # Create transforms
        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761),
            ),
        ]

        val_test_transform = train_transform
        # add augmentations
        if kwargs["augment"]:
            train_transform = [
                transforms.RandomCrop(32, padding=4, padding_mode="symmetric"),
                transforms.RandomHorizontalFlip(),
            ] + train_transform

        self.train_transform = transforms.Compose(train_transform)
        self.val_test_transform = transforms.Compose(val_test_transform)

    def prepare_data(self):
        # download data, train then test
        datasets.CIFAR100(self.data_dir, train=True, download=True)
        datasets.CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # we set up only relevant datamodules when stage is specified
        if stage == "fit" or stage is None:
            cifar100 = datasets.CIFAR100(
                self.data_dir,
                train=True,
                transform=self.train_transform,
            )
            self.train_dataset, self.val_dataset = random_split(
                cifar100,
                [45000, 5000],
                generator=torch.Generator().manual_seed(getattr(self, "seed", 42)),
            )
        if stage == "test" or stage is None:
            self.test_dataset = datasets.CIFAR100(
                self.data_dir,
                train=False,
                transform=self.val_test_transform,
            )

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return test_dataloader
