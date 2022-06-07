import torch
import numpy as np

import ckconv
from .mfn import MFNBase


#############################################
#       MFN with Random Fourier Features
##############################################
class RFFeaturesLayer(torch.nn.Module):
    """
    Random fourier feature layer used in RFMFNs.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_channels: int,
        input_scale: float,
    ):
        super().__init__()

        assert (
            hidden_channels % 2 == 0
        ), f"hidden_channels must be even. Current {hidden_channels}"
        linear_hidden_channels = hidden_channels // 2

        # Define type of linear
        Linear = getattr(ckconv.nn, f"Linear{data_dim}d")
        # Construct and initialize
        self.linear = Linear(data_dim, linear_hidden_channels)
        self.linear.weight.data *= input_scale  # TODO standardize name (omega_0? )
        self.linear.bias.data.uniform_(
            -np.pi, np.pi
        )  # TODO: This does not make much sense

    def forward(self, x):
        out = self.linear(x)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


class RandomFourierMFN(MFNBase):
    def __init__(
        self,
        data_dim: int,
        out_channels: int,
        hidden_channels: int,
        no_layers: int,
        bias: bool,
        input_scale: float = 256.0,
        weight_scale: float = 1.0,
    ):
        super().__init__(
            data_dim=data_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            no_layers=no_layers,
            weight_scale=weight_scale,
            bias=bias,
        )
        self.filters = torch.nn.ModuleList(
            [
                RFFeaturesLayer(
                    data_dim=data_dim,
                    hidden_channels=hidden_channels,
                    input_scale=input_scale
                    / np.sqrt(
                        no_layers + 1
                    ),  # TODO: What is the effect of this? Other init required?
                )
                for _ in range(no_layers + 1)
            ]
        )
