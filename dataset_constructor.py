# torch
import torch

# built-in
import os

# typing
from omegaconf import OmegaConf
import pytorch_lightning as pl

# datamodules
import datamodules


def construct_datamodule(
    cfg: OmegaConf,
) -> pl.LightningDataModule:

    # Define num_workers
    if cfg.no_workers == -1:
        cfg.no_workers = int(os.cpu_count() / 4)

    # Define pin_memory
    if torch.cuda.is_available() and cfg.device == "cuda":
        pin_memory = True
    else:
        pin_memory = False

    # Gather module from datamodules, create instance and return
    dataset_name = f"{cfg.dataset.name}DataModule"
    dataset = getattr(datamodules, dataset_name)
    datamodule = dataset(
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size // cfg.train.accumulate_grad_steps,
        test_batch_size=cfg.test.batch_size_multiplier * cfg.train.batch_size,
        data_type=cfg.dataset.data_type,
        num_workers=cfg.no_workers,
        pin_memory=pin_memory,
        augment=cfg.dataset.augment,
        **cfg.dataset.params,
    )
    # Assert if the datamodule has the parameters needed for the model creation
    assert hasattr(datamodule, "data_dim")
    assert hasattr(datamodule, "input_channels")
    assert hasattr(datamodule, "output_channels")
    assert hasattr(datamodule, "data_type")
    return datamodule
