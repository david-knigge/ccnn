import torch
import ckconv
import numpy as np
from .mlp import MLPBase


class RFNet(MLPBase):
    def __init__(
        self,
        data_dim: int,
        out_channels: int,
        hidden_channels: int,
        omega_0: float,
        no_layers: int,
        NonlinearType: torch.nn.Module,
        NormType: torch.nn.Module,
        weight_norm: bool,
        bias: bool,
        **kwargs,
    ):
        # Define type of linear
        LinearType = getattr(ckconv.nn, f"Linear{data_dim}d")

        # construct the hidden and out layers of the network
        super().__init__(
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            no_layers=no_layers,
            LinearType=LinearType,
            NonlinearType=NonlinearType,
            NormType=NormType,
            weight_norm=weight_norm,
            bias=bias,
        )

        # Construct input embedding
        self.input_layers = RandomFourierEmbedding(
            data_dim=data_dim,
            out_channels=hidden_channels,
            omega_0=omega_0,
            bias=bias,
        )


class RandomFourierEmbedding(torch.nn.Module):
    def __init__(
        self,
        data_dim: int,
        out_channels: int,
        omega_0: float,
        bias: int,
    ):
        super().__init__()

        assert (
            out_channels % 2 == 0
        ), f"out_channels must be even. Current {out_channels}"
        linear_out_channels = out_channels // 2

        # Define type of linear
        Linear = getattr(ckconv.nn, f"Linear{data_dim}d")
        self.linear = Linear(
            in_channels=data_dim,
            out_channels=linear_out_channels,
            bias=bias,
        )
        # Initialize:
        self.linear.weight.data.normal_(0.0, 2 * np.pi * omega_0)
        # Save params to self
        self.omega_0 = omega_0

    def forward(self, x):
        out = self.linear(x)
        return torch.cat([torch.cos(out), torch.sin(out)], dim=1)
