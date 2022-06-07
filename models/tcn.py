import torch
from functools import partial
from . import modules

# project
import ckconv

# typing
from omegaconf import OmegaConf


class TCNBase(torch.nn.Module):
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
        block_type = net_cfg.block.type
        block_width_factors = net_cfg.block_width_factors
        nonlinearity = net_cfg.nonlinearity

        # Define dropout_in
        self.dropout_in = torch.nn.Dropout(dropout_in)

        # Unpack conv_type
        conv_type = conv_cfg.type
        # Define partials for types of convs
        if conv_type != "Conv":
            ConvType = partial(
                getattr(ckconv.nn, conv_type),
                data_dim=data_dim,
                kernel_cfg=kernel_cfg,
                conv_cfg=conv_cfg,
                mask_cfg=mask_cfg,
            )
        else:
            ConvType = partial(
                getattr(ckconv.nn, f"Conv{data_dim}d"),
                kernel_size=int(kernel_cfg.size),
                padding=conv_cfg.padding,
                stride=conv_cfg.stride,
                bias=conv_cfg.bias,
            )
        # -------------------------

        # Define NormType
        if norm == "BatchNorm":
            norm_name = f"BatchNorm{data_dim}d"
        else:
            norm_name = norm
        # Select ckconv as base package if possible
        if hasattr(ckconv.nn, norm):
            lib = ckconv.nn
        else:
            lib = torch.nn
        NormType = getattr(lib, norm_name)

        # Define NonlinearType
        NonlinearType = getattr(torch.nn, nonlinearity)

        # Define LinearType
        LinearType = getattr(ckconv.nn, f"Linear{data_dim}d")

        # Create Blocks
        # -------------------------
        if block_type == "default":
            BlockType = modules.TCNBlock
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
                input_ch = in_channels
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
                    dropout=dropout,
                )
            )
            # if pool: # Pool is not used in our experiments
            #     raise NotImplementedError()
            #     # blocks.append(torch.nn.MaxPool1d(kernel_size=2))

        self.blocks = torch.nn.Sequential(*blocks)
        # -------------------------

        # Define Output Layers:
        # -------------------------
        # 1. Calculate output channels of blocks
        if block_width_factors[0] == 0.0:
            final_no_hidden = hidden_channels
        else:
            final_no_hidden = int(hidden_channels * block_width_factors[-2])
        # 2. instantiate & initialize last layer
        if block_type != "default":
            self.out_norm = NormType(final_no_hidden)
        else:
            self.out_norm = torch.nn.Identity()
        self.out_layer = LinearType(
            in_channels=final_no_hidden, out_channels=out_channels
        )
        torch.nn.init.kaiming_normal_(self.out_layer.weight)
        self.out_layer.bias.data.fill_(value=0.0)
        # -------------------------

        # Save variables in self
        self.data_dim = data_dim

    def forward(self, x):
        raise NotImplementedError


class TCN_sequence(TCNBase):
    def forward(self, x):
        # Dropout in
        x = self.dropout_in(x)
        # Blocks
        out = self.blocks(x)
        # Final layer on last sequence element
        out = self.out_norm(out)
        out = self.out_layer(out[:, :, -1:])
        return out.squeeze(-1)
