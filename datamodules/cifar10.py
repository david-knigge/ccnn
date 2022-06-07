import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl

import numpy as np

# config
from hydra import utils


class CIFAR10DataModule(pl.LightningDataModule):
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
        self.noise_padded = kwargs["noise_padded"]
        self.grayscale = kwargs["grayscale"]  # Used for LRA

        # Determine data_type
        if data_type == "default":
            self.data_type = "image"
            self.data_dim = 2
        elif data_type == "sequence":
            self.data_type = data_type
            self.data_dim = 1
        else:
            raise ValueError(f"data_type {data_type} not supported.")

        # Noise for noise-padded sCIFAR10
        if self.data_type == "sequence" and self.noise_padded:
            self.rands = torch.randn(1, 1000 - 32, 96)

        # Determine sizes of dataset
        if self.data_type == "sequence" and self.noise_padded:
            self.input_channels = 96
        elif self.grayscale:
            self.input_channels = 1
        else:
            self.input_channels = 3
        self.output_channels = 10

        # Create transforms
        if self.grayscale:
            train_transform = [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
            ]
        else:
            train_transform = [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.247, 0.243, 0.261),
                ),
            ]

        val_test_transform = train_transform
        # Augmentation before normalization, taken from:
        # https://github.com/dipuk0506/SpinalNet/blob/master/CIFAR-10/ResNet_default_and_SpinalFC_CIFAR10.py#L39
        if kwargs["augment"]:
            train_transform = [
                transforms.RandomCrop(32, padding=4, padding_mode="symmetric"),
                transforms.RandomHorizontalFlip(),
            ] + train_transform

        self.train_transform = transforms.Compose(train_transform)
        self.val_test_transform = transforms.Compose(val_test_transform)

    def prepare_data(self):
        # download data, train then test
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # we set up only relevant datamodules when stage is specified
        if stage == "fit" or stage is None:
            cifar10 = datasets.CIFAR10(
                self.data_dir,
                train=True,
                transform=self.train_transform,
            )
            self.train_dataset, self.val_dataset = random_split(
                cifar10,
                [45000, 5000],
                generator=torch.Generator().manual_seed(getattr(self, "seed", 42)),
            )
        if stage == "test" or stage is None:
            self.test_dataset = datasets.CIFAR10(
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

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if self.data_type == "sequence":
            # If sequential, flatten the input [B, C, Y, X] -> [B, C, -1]
            x, y = batch
            x_shape = x.shape
            # If noise padded, transform into a sequence of 96 channels and 1000 length
            if self.noise_padded:
                x = torch.cat(
                    (
                        x.permute(0, 2, 1, 3).reshape(x_shape[0], 32, 96),
                        self.rands.repeat(x_shape[0], 1, 1),
                    ),
                    dim=1,
                ).permute(0, 2, 1)
            else:
                x = x.view(x_shape[0], x_shape[1], -1)
            batch = x, y
        return batch
