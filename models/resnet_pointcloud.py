import torch
import torch_geometric as pyg
from functools import partial
from . import modules

# project
import ckconv

# typing
from omegaconf import OmegaConf


class ResNetPointCloudBase(torch.nn.Module):
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
        data_dim = net_cfg.data_dim
        norm = net_cfg.norm
        dropout = net_cfg.dropout
        dropout_in = net_cfg.dropout_in
        dropout_type = net_cfg.dropout_type
        block_type = net_cfg.block.type
        block_prenorm = net_cfg.block.prenorm
        block_width_factors = net_cfg.block_width_factors
        downsampling = net_cfg.downsampling
        downsampling_size = net_cfg.downsampling_size
        nonlinearity = net_cfg.nonlinearity

        # Define dropout_in
        self.dropout_in = torch.nn.Dropout(dropout_in)

        # Unpack conv_type
        conv_type = conv_cfg.type
        # Define partials for types of convs
        ConvType = partial(
            getattr(ckconv.nn, conv_type),
            data_dim=data_dim,
            kernel_cfg=kernel_cfg,
            conv_cfg=conv_cfg,
            mask_cfg=mask_cfg,
        )
        # -------------------------

        # Define NormType
        assert norm == "BatchNorm"
        NormType = getattr(ckconv.nn, f"Graph{norm}")

        # Define NonlinearType
        assert nonlinearity == "GELU"
        NonlinearType = getattr(ckconv.nn, f"Graph{nonlinearity}")

        # Define LinearType
        LinearType = getattr(ckconv.nn, "GraphLinear")

        # Define DownsamplingType
        DownsamplingType = getattr(torch.nn, f"MaxPool{data_dim}d")

        # Define Dropout layer type
        DropoutType = getattr(ckconv.nn, f"Graph{dropout_type}")

        # Create Input Layers
        self.conv1 = ConvType(in_channels=in_channels, out_channels=hidden_channels)
        self.norm1 = NormType(hidden_channels)
        self.nonlinear = NonlinearType()

        # Create Blocks
        # -------------------------
        if block_type == "default":
            BlockType = modules.ResNetBlock
        else:
            BlockType = getattr(modules, f"{block_type}Block")
        # 1. Create vector of width_factors:
        # If value is zero, then all values are one
        if block_width_factors[0] == 0.0:
            width_factors = (1,) * no_blocks
        else:
            width_factors = [
                (factor,) * n_blcks
                for factor, n_blcks in ckconv.utils.pairwise_iterable(
                    block_width_factors
                )
            ]
            width_factors = [
                factor for factor_tuple in width_factors for factor in factor_tuple
            ]
        if len(width_factors) != no_blocks:
            raise ValueError(
                "The size of the width_factors does not matched the number of blocks in the network."
            )
        # 2. Create blocks
        blocks = []
        for i in range(no_blocks):
            print(f"Block {i}/{no_blocks}")

            if i == 0:
                input_ch = hidden_channels
                hidden_ch = int(hidden_channels * width_factors[i])
            else:
                input_ch = int(hidden_channels * width_factors[i - 1])
                hidden_ch = int(hidden_channels * width_factors[i])

            blocks.append(
                BlockType(
                    in_channels=input_ch,
                    out_channels=hidden_ch,
                    ConvType=ConvType,
                    NonlinearType=NonlinearType,
                    NormType=NormType,
                    LinearType=LinearType,
                    DropoutType=DropoutType,
                    dropout=dropout,
                    prenorm=block_prenorm,
                )
            )

            # Check whether we need to add a downsampling block here.
            if i in downsampling:
                blocks.append(DownsamplingType(kernel_size=downsampling_size))

        # TODO pyg sequential.
        self.blocks = torch.nn.Sequential(*blocks)
        # -------------------------

        # Define Output Layers:
        # -------------------------
        # 1. Calculate output channels of blocks
        if block_width_factors[0] == 0.0:
            final_no_hidden = hidden_channels
        else:
            final_no_hidden = int(hidden_channels * block_width_factors[-2])
        # 2. instantiate last layer
        self.out_layer = LinearType(
            in_channels=final_no_hidden, out_channels=out_channels
        )
        # 3. Initialize finallyr
        torch.nn.init.kaiming_normal_(self.out_layer.weight)
        self.out_layer.bias.data.fill_(value=0.0)
        # -------------------------
        if block_type == "S4" and block_prenorm:
            self.out_norm = NormType(final_no_hidden)
        else:
            self.out_norm = torch.nn.Identity()

        # Save variables in self
        self.data_dim = data_dim

    def forward(self, x):
        raise NotImplementedError


class ResNet_pointcloud(ResNetPointCloudBase):
    def forward(self, data):

        # Dropout in.
        data.x = self.dropout_in(data.x)

        # First layer.
        data = self.conv1(data)
        data = self.norm1(data)
        data = self.nonlinear(data)

        # Blockboys.
        data = self.blocks(data)
        # Final layer on last sequence element
        data = self.out_norm(data)

        # Pool over spatial dims
        data.x = torch.mean(data.x, dim=1)
        # Final layer
        data = self.out_layer(data)
        return data.x
