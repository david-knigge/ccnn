import torch
import numpy as np

import ckconv
from .mfn import MFNBase, gaussian_window


class MAGNetLayer(torch.nn.Module):
    """
    Gabor-like filter as used in GaborNet.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_channels: int,
        steerable: bool,
        omega_0: float,
        alpha: float,
        beta: float,
        bias: bool,
        init_spatial_value: float,
        causal: bool,
    ):
        super().__init__()

        # Define type of linear to use
        Linear = getattr(ckconv.nn, f"Linear{data_dim}d")

        # # Define type of linear to use, for regular grids we can use 1x1 convolution operations to speed up computation.
        # if not irregular:
        #     Linear = getattr(ckconv.nn, f"Linear{data_dim}d")
        #
        # # For irregular data we need conventional linear layers.
        # else:
        #     Linear = getattr(torch.nn, f"Linear{data_dim}d")

        # Construct & initialize parameters
        # """
        # Background:
        # If we use a 2D mask, we must initialize the mask around 0. If we use a 1D
        # mask, however, we initialize the mean to 1. That is, the last sequence
        # element. As a result, we must also recenter the mu-values around 1.0.
        # In addition, the elements at the positive size are not used at the beginning.
        # Hence, we are only interested in elements on the negative size of the line.
        # """
        if causal and init_spatial_value != 1.0:
            mu = 1.0 - init_spatial_value * torch.rand(hidden_channels, data_dim)
        else:
            mu = init_spatial_value * (2 * torch.rand(hidden_channels, data_dim) - 1)
        self.mu = torch.nn.Parameter(mu)
        self.gamma = torch.nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample(
                (hidden_channels, data_dim)
            )
        )
        self.linear = Linear(data_dim, hidden_channels, bias=bias)
        self.linear.weight.data *= (
            2 * np.pi * omega_0 * self.gamma.view(*self.gamma.shape, *((1,) * data_dim))
        )
        self.linear.bias.data.fill_(0.0)
        # If steerable, create thetas
        self.steerable = steerable
        if self.steerable:
            self.theta = torch.nn.Parameter(torch.rand(hidden_channels))
        # Save parameters in self.
        self.data_dim = data_dim
        self.omega_0 = omega_0

    def forward(self, x):
        if self.steerable:
            gauss_window = rotated_gaussian_window(
                x,
                self.gamma.view(1, *self.gamma.shape, *((1,) * self.data_dim)),
                self.theta,
                self.mu.view(1, *self.mu.shape, *((1,) * self.data_dim)),
            )
        else:
            gauss_window = gaussian_window(
                x,
                self.gamma.view(1, *self.gamma.shape, *((1,) * self.data_dim)),
                self.mu.view(1, *self.mu.shape, *((1,) * self.data_dim)),
            )
        return gauss_window * torch.sin(self.linear(x))


def rotation_matrix(theta):
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    return torch.stack([cos, sin, -sin, cos], dim=-1).view(-1, 2, 2)


class MAGNet(MFNBase):
    def __init__(
        self,
        data_dim: int,
        hidden_channels: int,
        out_channels: int,
        no_layers: int,
        steerable: bool,
        bias: bool,
        causal: bool,
        omega_0: float,
        alpha: float = 6.0,
        beta: float = 1.0,
        init_spatial_value: float = 1.0,
        **kwargs,
    ):
        """
        Multiplicative Anisotropic Gabor Network.
        :param dim_linear: Dimensionality of input signal, e.g. 2 for images.
        :param hidden_channels: Amount of hidden channels to use.
        :param out_channels: Amount of output channels to use.
        :param no_layers: Amount of layers to use in kernel generator.
        :param steerable: Whether to learn steerable kernels.
        :param bias: Whether to use bias.
        :param bias_init: Bias init strategy.
        :param input_scale: Scaling factor for linear functions.
        :param weight_scale: Scale for uniform weight initialization of linear
            layers.
        :param alpha: Base alpha for Gamma distribution to initialize Gabor
            filter variance. This value is divided by `layer_idx+1`.
        :param beta: Beta for Gamma distribution to initialize Gabor filter
            variance.
        :param init_spatial_value: Initial mu for gabor filters.
        """
        super().__init__(
            data_dim=data_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            no_layers=no_layers,
            bias=bias,
        )
        self.filters = torch.nn.ModuleList(
            [
                MAGNetLayer(
                    data_dim=data_dim,
                    hidden_channels=hidden_channels,
                    steerable=steerable,
                    omega_0=omega_0,
                    alpha=alpha / (layer + 1),
                    beta=beta,
                    bias=bias,
                    init_spatial_value=init_spatial_value,
                    causal=causal,
                )
                for layer in range(no_layers + 1)
            ]
        )
        # Initialize
        for idx, lin in enumerate(self.linears):
            torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity="linear")
            if lin.bias is not None:
                lin.bias.data.fill_(1.0)
        torch.nn.init.kaiming_uniform_(self.output_linear.weight, nonlinearity="linear")
        self.output_linear.bias.data.fill_(0.0)


def rotate(theta, input):
    # theta.shape = [Out, 1]
    # input.shape = [B, Channels, 2, X, Y]
    return torch.einsum("coi, bcixy -> bcoxy", rotation_matrix(theta), input)


def rotated_gaussian_window(x, gamma, theta, mu):
    return torch.exp(
        -0.5 * ((gamma * rotate(2 * np.pi * theta, x.unsqueeze(1) - mu)) ** 2).sum(2)
    )


if __name__ == "__main__":
    data_dim = 3
    # Create MAGNet
    magnet = MAGNet(
        data_dim=data_dim,
        hidden_channels=32,
        out_channels=50,
        no_layers=3,
        steerable=False,
        bias=True,
        causal=False,
        omega_0=100.0,
    )

    coords = ckconv.utils.linspace_grid(
        [
            20,
        ]
        * data_dim
    ).unsqueeze(0)

    out = magnet(coords)
    print(out.shape)
