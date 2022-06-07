import torch
import ckconv
import numpy as np


class MFNBase(torch.nn.Module):
    """
    Multiplicative filter network base class.
    Expects the child class to define the 'filters' attribute, which should be
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_channels: int,
        out_channels: int,
        no_layers: int,
        bias: bool,
    ):
        super().__init__()

        # Define type of linear
        Linear = getattr(ckconv.nn, f"Linear{data_dim}d")
        # Hidden layers
        self.linears = torch.nn.ModuleList(
            [
                Linear(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    bias=bias,
                )
                for _ in range(no_layers)
            ]
        )
        # Final layer
        self.output_linear = Linear(
            in_channels=hidden_channels,
            out_channels=out_channels,
            bias=bias,
        )

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linears[i - 1](out)
        out = self.output_linear(out)
        return out


#############################################
#       FourierNet
##############################################
class FourierLayer(torch.nn.Module):
    """
    Sine filter as used in FourierNet.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_channels: int,
        omega_0: float,
    ):
        super().__init__()

        # Define type of linear
        Linear = getattr(ckconv.nn, f"Linear{data_dim}d")
        # Construct linear
        self.linear = Linear(data_dim, hidden_channels)
        # Initialize
        w_std = 1.0 / self.linear.weight.shape[1]  # Following Sitzmann et al. 2020
        self.linear.weight.data.uniform_(-w_std, w_std)
        self.linear.weight.data *= 2.0 * np.pi * omega_0
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0.0)
        # Save params in self
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.linear(x))

    def extra_repr(self):
        return f"omega_0={self.omega_0}"


class FourierNet(MFNBase):
    def __init__(
        self,
        data_dim: int,
        hidden_channels: int,
        out_channels: int,
        no_layers: int,
        bias: bool,
        omega_0: float,
        **kwargs,
    ):
        super().__init__(
            data_dim=data_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            no_layers=no_layers,
            bias=bias,
        )
        self.filters = torch.nn.ModuleList(
            [
                FourierLayer(
                    data_dim=data_dim,
                    hidden_channels=hidden_channels,
                    omega_0=omega_0,
                )
                for _ in range(no_layers + 1)
            ]
        )
        # Initialize
        var_g = 0.5
        with torch.no_grad():
            # Init so that all freq. components have the same amplitude on initialization
            accumulated_weight = None
            for idx, lin in enumerate(self.linears):
                layer = idx + 1
                torch.nn.init.orthogonal_(lin.weight)
                lin.weight.data *= np.sqrt(1.0 / var_g)

                # Get norm of weights so far.
                if accumulated_weight is None:
                    accumulated_weight = lin.weight.data.clone()
                else:
                    accumulated_weight = torch.einsum(
                        "ab...,bc...->ac...", lin.weight.data, accumulated_weight
                    )
                accumulated_value = accumulated_weight.view(
                    accumulated_weight.shape[0], -1
                ).sum(dim=-1)

                # Initialize the bias
                if lin.bias is not None:
                    lin.bias.data = (
                        accumulated_value / (hidden_channels * 2.0**layer)
                    ).flatten()

        torch.nn.init.kaiming_uniform_(self.output_linear.weight, nonlinearity="linear")
        self.output_linear.bias.data.fill_(0.0)


#############################################
#       GaborNet
##############################################
class GaborNet(MFNBase):
    def __init__(
        self,
        data_dim: int,
        hidden_channels: int,
        out_channels: int,
        no_layers: int,
        bias: bool,
        omega_0: float,
        alpha: float = 6.0,
        beta: float = 1.0,
        init_spatial_value: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            data_dim=data_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            no_layers=no_layers,
            bias=bias,
        )
        self.filters = torch.nn.ModuleList(
            [
                GaborLayer(
                    data_dim=data_dim,
                    hidden_channels=hidden_channels,
                    omega_0=omega_0,
                    alpha=alpha / (no_layers + 1),
                    beta=beta,
                    bias=bias,
                    init_spatial_value=init_spatial_value,
                )
                for _ in range(no_layers + 1)
            ]
        )
        # Initialize
        for idx, lin in enumerate(self.linears):
            torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity="linear")
            if lin.bias is not None:
                lin.bias.data.fill_(1.0)
        torch.nn.init.kaiming_uniform_(self.output_linear.weight, nonlinearity="linear")
        self.output_linear.bias.data.fill_(0.0)


class GaborLayer(torch.nn.Module):
    """
    Gabor-like filter as used in GaborNet.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_channels: int,
        omega_0: float,
        alpha: float,
        beta: float,
        bias: bool,
        init_spatial_value: float,
    ):
        super().__init__()

        # Define type of linear
        Linear = getattr(ckconv.nn, f"Linear{data_dim}d")

        # Construct & initialize parameters
        # """
        # Background:
        # If we use a 2D mask, we must initialize the mask around 0. If we use a 1D
        # mask, however, we initialize the mean to 1. That is, the last sequence
        # element. As a result, we must also recenter the mu-values around 1.0.
        # In addition, the elements at the positive size are not used at the beginning.
        # Hence, we are only interested in elements on the negative size of the line.
        # """
        if data_dim == 1 and init_spatial_value != 1.0:
            mu = 1.0 - init_spatial_value * torch.rand(hidden_channels, data_dim)
        else:
            mu = init_spatial_value * (2 * torch.rand(hidden_channels, data_dim) - 1)
        self.mu = torch.nn.Parameter(mu)
        self.gamma = torch.nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample(
                (hidden_channels, 1)
            )  # Isotropic
        )

        # Create and initialize parameters
        self.linear = Linear(data_dim, hidden_channels, bias=bias)
        # Initialize
        w_std = 1.0 / self.linear.weight.shape[1]  # Following Sitzmann et al. 2020
        self.linear.weight.data.uniform_(-w_std, w_std)
        self.linear.weight.data *= (
            2.0
            * np.pi
            * omega_0
            * self.gamma.view(*self.gamma.shape, *((1,) * data_dim))
        )
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0.0)

        # Save params. in self
        self.data_dim = data_dim

    def forward(self, x):
        gauss_window = gaussian_window(
            x,
            self.gamma.view(
                1, *self.gamma.shape, *((1,) * self.data_dim)
            ),  # TODO. We can avoid doing this
            self.mu.view(1, *self.mu.shape, *((1,) * self.data_dim)),
        )
        return gauss_window * torch.sin(self.linear(x))


def gaussian_window(x, gamma, mu):
    return torch.exp(-0.5 * ((gamma * (x.unsqueeze(1) - mu)) ** 2).sum(2))
