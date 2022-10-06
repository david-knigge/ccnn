"""
Adapted from
https://github.com/dwromero/ckconv/blob/master/datasets/speech_commands.py
which is adapted from
https://github.com/patrick-kidger/NeuralCDE/blob/758d3a7134e3a691013e5cc6b7f68f277e9e6b69/experiments/datasets/speech_commands.py
"""
import os
import pathlib
import urllib.request
import tarfile

import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchaudio
from hydra import utils

from .utils import normalise_data, split_data, load_data_from_partition, save_data


class SpeechCommandsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        test_batch_size,
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
        self.mfcc = kwargs["mfcc"]
        self.drop_rate = kwargs["drop_rate"]

        root = pathlib.Path(self.data_dir)
        self.download_location = root / "SpeechCommands"
        self.data_processed_location = root / "SpeechCommands" / "processed_data"
        if self.mfcc:
            self.data_processed_location = self.data_processed_location / "mfcc"
        else:
            self.data_processed_location = self.data_processed_location / "raw"
            if self.drop_rate != 0:
                self.data_processed_location = pathlib.Path(
                    f"{self.data_processed_location}_dropped{self.drop_rate}"
                )

        # set data type & data dim
        self.data_type = "sequence"
        self.data_dim = 1
        # Determine sizes of dataset
        if self.mfcc:
            self.input_channels = 20
        else:
            self.input_channels = 1
        self.output_channels = 10

    def prepare_data(self):
        # TODO: Make this more robust
        if not os.path.exists(self.data_processed_location / "train_X.pt"):
            # download data
            self._download()
            # create train, val & test datamodules
            train_X, val_X, test_X, train_y, val_y, test_y = self._process_data()
            # save to disk
            save_data(
                self.data_processed_location,
                train_X=train_X,
                val_X=val_X,
                test_X=test_X,
                train_y=train_y,
                val_y=val_y,
                test_y=test_y,
            )

    def setup(self, stage=None):
        # we set up only relevant datamodules when stage is specified
        if stage == "fit" or stage is None:
            # train
            X_train, y_train = load_data_from_partition(
                self.data_processed_location, partition="train"
            )
            self.train_dataset = TensorDataset(X_train, y_train)
            # validation
            X_val, y_val = load_data_from_partition(
                self.data_processed_location, partition="val"
            )
            self.val_dataset = TensorDataset(X_val, y_val)
        if stage == "test" or stage is None:
            # test
            X_test, y_test = load_data_from_partition(
                self.data_processed_location, partition="test"
            )
            self.test_dataset = TensorDataset(X_test, y_test)

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

    # Added routines
    def _download(self):
        download_file_path = self.download_location / "speech_commands.tar.gz"
        if os.path.exists(download_file_path):
            return
        if not os.path.exists(self.download_location):
            os.mkdir(self.download_location)
        urllib.request.urlretrieve(
            "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
            download_file_path,
        )
        with tarfile.open(download_file_path, "r") as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, self.download_location)

    def _process_data(self):
        if not os.path.exists(self.data_processed_location):
            os.makedirs(self.data_processed_location)

        X = torch.empty(34975, 16000, 1)
        y = torch.empty(34975, dtype=torch.long)

        batch_index = 0
        y_index = 0
        for foldername in (
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
        ):
            loc = self.download_location / foldername
            for filename in os.listdir(loc):
                audio, _ = torchaudio.load(
                    loc / filename,
                    channels_first=False,
                )  # for forward compatbility if they fix it
                audio = (
                    audio / 2**15
                )  # Normalization argument doesn't seem to work so we do it manually.

                # A few samples are shorter than the full length; for simplicity we discard them.
                if len(audio) != 16000:
                    continue

                X[batch_index] = audio
                y[batch_index] = y_index
                batch_index += 1
            y_index += 1
        assert batch_index == 34975, "batch_index is {}".format(batch_index)

        # If MFCC, then we compute these coefficients.
        if self.mfcc:
            X = torchaudio.transforms.MFCC(
                log_mels=True, n_mfcc=20, melkwargs=dict(n_fft=200, n_mels=64)
            )(X.squeeze(-1)).detach()
            # X is of shape (batch=34975, channels=20, length=161)
        else:
            X = X.unsqueeze(1).squeeze(-1)
            # X is of shape (batch=34975, channels=1, length=16000)

        # If dropped is different than zero, randomly drop that quantity of data from the dataset.
        if self.drop_rate != 0:
            generator = torch.Generator().manual_seed(56789)
            X_removed = []
            for Xi in X:
                removed_points = (
                    torch.randperm(X.shape[-1], generator=generator)[
                        : int(X.shape[-1] * float(self.drop_rate) / 100.0)
                    ]
                    .sort()
                    .values
                )
                Xi_removed = Xi.clone()
                Xi_removed[:, removed_points] = float("nan")
                X_removed.append(Xi_removed)
            X = torch.stack(X_removed, dim=0)

        # Normalize data
        if self.mfcc:
            X = normalise_data(X.transpose(1, 2), y).transpose(1, 2)
        else:
            X = normalise_data(X, y)

        # Once the data is normalized append times and mask values if required.
        if self.drop_rate != 0:
            # Get mask of possitions that are deleted
            mask_exists = (~torch.isnan(X[:, :1, :])).float()
            X = torch.where(~torch.isnan(X), X, torch.Tensor([0.0]))
            X = torch.cat([X, mask_exists], dim=1)

        train_X, val_X, test_X = split_data(X, y)
        train_y, val_y, test_y = split_data(y, y)

        return (
            train_X,
            val_X,
            test_X,
            train_y,
            val_y,
            test_y,
        )
