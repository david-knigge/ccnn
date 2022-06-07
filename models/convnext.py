import torch
from torch.nn import functional

from functools import partial
from .modules.convnext_block import ConvNeXtBlock

from copy import deepcopy

# project
import ckconv

# typing
from omegaconf import OmegaConf


class ConvNeXtBase(torch.nn.Module):
    """ConvNeXt implementation. Based on 'A ConvNet for the 2020s' by Lui et al.

    With code adapted from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        net_cfg: OmegaConf,
        kernel_cfg: OmegaConf,
        conv_cfg: OmegaConf,
        mask_cfg: OmegaConf,
    ):
        super().__init__()

        # Unpack arguments from net_config
        hidden_channels = net_cfg.no_hidden
        no_blocks = net_cfg.no_blocks

        # the original resnet and convnext architectures define multiple feature resolution stages
        no_stages = net_cfg.no_stages if hasattr(net_cfg, "no_stages") else 1

        data_dim = net_cfg.data_dim
        norm = net_cfg.norm
        nonlinearity = net_cfg.nonlinearity

        # Unpack conv_type
        conv_type = conv_cfg.type

        # The stem cell has larger stride and kernel size
        stem_cfg = deepcopy(conv_cfg)
        stem_cfg.size = 15
        stem_cfg.stride = 15
        stem_cfg.padding = 0
        stem_kernel_cfg = deepcopy(kernel_cfg)
        stem_kernel_cfg.size = stem_cfg.size

        # The downsampling kernels also have different stride and kernel size
        downsampling_cfg = deepcopy(conv_cfg)
        downsampling_cfg.size = 15
        downsampling_cfg.stride = 15
        downsampling_cfg.padding = 0
        downsampling_kernel_cfg = deepcopy(kernel_cfg)
        downsampling_kernel_cfg.size = downsampling_cfg.size

        depthwise_cfg = deepcopy(conv_cfg)
        if type(depthwise_cfg.size) != str:
            depthwise_cfg.padding = depthwise_cfg.size // 2
        depthwise_kernel_cfg = deepcopy(kernel_cfg)
        depthwise_kernel_cfg.size = depthwise_cfg.size

        # Define partials for types of convs
        if conv_type == "CKConv":
            StemConv = partial(
                ckconv.nn.CKConv,
                data_dim=data_dim,
                kernel_cfg=stem_kernel_cfg,
                conv_cfg=stem_cfg,
            )
            DownSamplingConv = partial(
                ckconv.nn.CKConv,
                data_dim=data_dim,
                kernel_cfg=downsampling_kernel_cfg,
                conv_cfg=downsampling_cfg,
            )
            DepthwiseConv = partial(
                ckconv.nn.CKConv,
                data_dim=data_dim,
                kernel_cfg=downsampling_kernel_cfg,
                conv_cfg=depthwise_cfg,
            )

            # Implement the pointwise convolution using regular conv operation (as we only sample a single spatial
            # location)
            PointwiseConv = partial(
                getattr(torch.nn, f"Conv{data_dim}d"),
                kernel_size=1,
                padding=0,
                stride=1,
                bias=conv_cfg.bias,
            )
        elif conv_type == "Conv":
            StemConv = partial(
                getattr(torch.nn, f"Conv{data_dim}d"),
                kernel_size=int(stem_cfg.size),
                padding=stem_cfg.padding,
                stride=stem_cfg.stride,
                bias=stem_cfg.bias,
            )
            DownSamplingConv = partial(
                getattr(torch.nn, f"Conv{data_dim}d"),
                kernel_size=int(downsampling_cfg.size),
                padding=downsampling_cfg.padding,
                stride=downsampling_cfg.stride,
                bias=downsampling_cfg.bias,
            )
            DepthwiseConv = partial(
                getattr(torch.nn, f"Conv{data_dim}d"),
                kernel_size=int(depthwise_cfg.size),
                padding=depthwise_cfg.padding,
                stride=depthwise_cfg.stride,
                bias=depthwise_cfg.bias,
            )
            PointwiseConv = partial(
                getattr(torch.nn, f"Conv{data_dim}d"),
                kernel_size=1,
                padding=0,
                stride=1,
                bias=conv_cfg.bias,
            )
        else:
            raise NotImplementedError(f"conv_type = {conv_type}")
        # -------------------------

        # Define NormType
        if norm == "BatchNorm":
            norm_name = f"BatchNorm{data_dim}d"
        else:
            norm_name = norm
        if hasattr(ckconv.nn, norm):
            lib = ckconv.nn
        else:
            lib = torch.nn
        NormType = getattr(lib, norm_name)

        # Define NonlinearType
        NonlinearType = getattr(torch.nn, nonlinearity)

        # Define LinearType
        LinearType = getattr(ckconv.nn, f"Linear{data_dim}d")

        # define stem and downsampling layers
        self.downsample_layers = torch.nn.ModuleList()

        # Create Stem Cell
        # -------------------------
        # stem cell: 4x4 stride 4 convolution followed by normalisation
        self.stem = torch.nn.Sequential(
            StemConv(
                in_channels=in_channels,
                out_channels=hidden_channels,
            ),
            NormType(hidden_channels),
        )

        # Create ConvNeXt Stages
        # -------------------------
        # Each stage consists of a number of ConvNeXt blocks, followed by a normalisation and downsampling
        self.stages = torch.nn.ModuleList()
        for i in range(no_stages):

            # Downsampling should happen after every stage except the last
            if i > 0:

                # For now, we fix the downsampling rate
                self.stages.append(
                    torch.nn.Sequential(
                        NormType(hidden_channels),
                        DownSamplingConv(
                            in_channels=hidden_channels, out_channels=hidden_channels
                        ),
                    )
                )

            for j in range(no_blocks):
                self.stages.append(
                    ConvNeXtBlock(
                        channels=hidden_channels,
                        DepthwiseConv=DepthwiseConv,
                        PointwiseConv=PointwiseConv,
                        NonlinearType=NonlinearType,
                        NormType=NormType,
                        LinearType=LinearType,
                    )
                )

        # Define Output Layers:
        # -------------------------
        #
        self.norm = torch.nn.LayerNorm(
            hidden_channels, eps=1e-6
        )  # final norm layer TODO make dynamic
        self.head = torch.nn.Linear(
            hidden_channels, out_channels
        )  # TODO bias to zero at init, implement as linear1d

    def forward(self, x):
        raise NotImplementedError


class ConvNeXt_sequence(ConvNeXtBase):
    def forward(self, x):
        x = self.stem(x)
        for mod in self.stages:
            x = mod(x)

        x = self.norm(x)

        # Take the last spatial location (causal conv)
        x = x[:, :, -1:]
        x = self.head(x)
        return x
