# torch
import torch
import pytorch_lightning as pl

# project
import ckconv
import models
from models.lightning_wrappers import (
    ClassificationWrapper,
    RegressionWrapper,
)

# typing
from omegaconf import OmegaConf


def construct_model(
    cfg: OmegaConf,
    datamodule: pl.LightningDataModule,
):
    """
    :param cfg: configuration file
    :return: An instance of torch.nn.Module
    """
    # Get parameters of model from task type
    data_dim = datamodule.data_dim
    in_channels = datamodule.input_channels
    out_channels = datamodule.output_channels
    data_type = datamodule.data_type

    # Get type of model from task type
    net_type = f"{cfg.net.type}_{data_type}"

    # Overwrite data_dim in cfg.net
    cfg.net.data_dim = data_dim

    # Print automatically derived model parameters.
    print(
        f"Automatic Parameters:\n dataset = {cfg.dataset.name}, "
        f"net_name = {net_type},"
        f" data_dim = {data_dim}"
        f" in_channels = {in_channels},"
        f" out_chanels = {out_channels}."
    )
    if out_channels == 2:
        print(
            "The model will output one single channel. We use BCEWithLogitsLoss for training."
        )

    # Create and return model
    net_type = getattr(models, net_type)
    network = net_type(
        in_channels=in_channels,
        out_channels=out_channels if out_channels != 2 else 1,
        net_cfg=cfg.net,
        kernel_cfg=cfg.kernel,
        conv_cfg=cfg.conv,
        mask_cfg=cfg.mask,
    )

    # Wrap in PytorchLightning
    Wrapper = ClassificationWrapper
    model = Wrapper(
        network=network,
        cfg=cfg,
    )
    # return model
    return model
