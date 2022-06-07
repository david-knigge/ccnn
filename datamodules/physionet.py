"""
Adapted from https://github.com/dwromero/ckconv/blob/master/datasets/physionet.py
which is adapted from
https://github.com/patrick-kidger/NeuralCDE/blob/758d3a7134e3a691013e5cc6b7f68f277e9e6b69/experiments/datasets/speech_commands.py
"""
import os
import pathlib
import urllib.request
import zipfile
import csv
import math

import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
from hydra import utils

from .utils import normalise_data, split_data, load_data_from_partition, save_data


class PhysioNetDataModule(pl.LightningDataModule):
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
        self.download_location = root / "sepsis"
        self.data_processed_location = root / "sepsis" / "processed_data"

        # set data type & data dim
        self.data_type = "sequence"
        self.data_dim = 1
        # Determine sizes of dataset
        self.input_channels = 75
        self.output_channels = 1

    def prepare_data(self):
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
        loc_Azip = self.download_location / "training_setA.zip"
        loc_Bzip = self.download_location / "training_setB.zip"

        if not os.path.exists(loc_Azip):

            if not os.path.exists(self.download_location):
                os.mkdir(self.download_location)
            urllib.request.urlretrieve(
                "https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip",
                str(loc_Azip),
            )
            urllib.request.urlretrieve(
                "https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip",
                str(loc_Bzip),
            )

            with zipfile.ZipFile(loc_Azip, "r") as f:
                f.extractall(str(self.download_location))
            with zipfile.ZipFile(loc_Bzip, "r") as f:
                f.extractall(str(self.download_location))
            for folder in ("training", "training_setB"):
                for filename in os.listdir(self.download_location / folder):
                    if os.path.exists(self.download_location / filename):
                        raise RuntimeError
                    os.rename(
                        self.download_location / folder / filename,
                        self.download_location / filename,
                    )

    def _process_data(self):
        X_times = []
        X_static = []
        y = []
        for filename in os.listdir(self.download_location):
            if filename.endswith(".psv"):
                with open(self.download_location / filename) as file:
                    time = []
                    label = 0.0
                    reader = csv.reader(file, delimiter="|")
                    reader = iter(reader)
                    next(reader)  # first line is headings
                    prev_iculos = 0
                    for line in reader:
                        assert len(line) == 41
                        (
                            *time_values,
                            age,
                            gender,
                            unit1,
                            unit2,
                            hospadmtime,
                            iculos,
                            sepsislabel,
                        ) = line
                        iculos = int(iculos)
                        if iculos > 72:  # keep at most the first three days
                            break
                        for iculos_ in range(prev_iculos + 1, iculos):
                            time.append([float("nan") for value in time_values])
                        prev_iculos = iculos
                        time.append([float(value) for value in time_values])
                        label = max(label, float(sepsislabel))
                    unit1 = float(unit1)
                    unit2 = float(unit2)
                    unit1_obs = not math.isnan(unit1)
                    unit2_obs = not math.isnan(unit2)
                    if not unit1_obs:
                        unit1 = 0.0
                    if not unit2_obs:
                        unit2 = 0.0
                    hospadmtime = float(hospadmtime)
                    if math.isnan(hospadmtime):
                        hospadmtime = 0.0  # this only happens for one record
                    static = [float(age), float(gender), unit1, unit2, hospadmtime]
                    static += [unit1_obs, unit2_obs]
                    if len(time) > 2:
                        X_times.append(time)
                        X_static.append(static)
                        y.append(label)
        final_indices = []
        for time in X_times:
            final_indices.append(len(time) - 1)
        maxlen = max(final_indices) + 1
        for time in X_times:
            for _ in range(maxlen - len(time)):
                time.append([float("nan") for value in time_values])

        X_times = torch.tensor(X_times)
        X_static = torch.tensor(X_static)
        y = torch.tensor(y).long()

        # Normalize data
        X_times = normalise_data(X_times, y)

        # Append extra channels together.
        augmented_X_times = []
        intensity = ~torch.isnan(X_times)  # of size (batch, stream, channels)
        intensity = intensity.to(X_times.dtype).cumsum(dim=1)
        augmented_X_times.append(intensity)
        augmented_X_times.append(X_times)
        X_times = torch.cat(augmented_X_times, dim=2)

        X_times = torch.where(~torch.isnan(X_times), X_times, torch.Tensor([0.0]))

        train_X_times, val_X_times, test_X_times = split_data(X_times, y)
        train_y, val_y, test_y = split_data(y, y)

        X_static_ = X_static[:, :-2]
        X_static_ = normalise_data(X_static_, y)
        X_static = (
            torch.cat([X_static_, X_static[:, -2:]], dim=1)
            .unsqueeze(1)
            .repeat(1, X_times.shape[1], 1)
        )

        train_X_static, val_X_static, test_X_static = split_data(X_static, y)

        # Concatenate
        train_X = torch.cat([train_X_times, train_X_static], dim=-1).transpose(-2, -1)
        val_X = torch.cat([val_X_times, val_X_static], dim=-1).transpose(-2, -1)
        test_X = torch.cat([test_X_times, test_X_static], dim=-1).transpose(-2, -1)

        return (
            train_X,
            val_X,
            test_X,
            train_y,
            val_y,
            test_y,
        )
