import torch
import ckconv
from torch.nn.utils import weight_norm as w_norm

# Based on https://github.com/lucidrains/siren-pytorch
from math import sqrt


class SIRENBase(torch.nn.Module):
    def __init__(
        self,
        data_dim: int,
        out_channels: int,
        hidden_channels: int,
        weight_norm: bool,
        no_layers: int,
        bias: bool,
        omega_0: float,
        learn_omega_0: bool,
        Linear_hidden: torch.nn.Module,
        Linear_out: torch.nn.Module,
    ):
        super().__init__()

        ActivationFunction = ckconv.nn.Sine

        # Construct the network
        # ---------------------
        # 1st layer:
        kernel_net = [
            Linear_hidden(
                in_channels=data_dim,
                out_channels=hidden_channels,
                omega_0=omega_0,
                learn_omega_0=learn_omega_0,
                bias=bias,
            ),
            ActivationFunction(),
        ]

        # Hidden layers:
        for _ in range(no_layers - 2):
            kernel_net.extend(
                [
                    Linear_hidden(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        omega_0=omega_0,
                        learn_omega_0=learn_omega_0,
                        bias=bias,
                    ),
                    ActivationFunction(),
                ]
            )
        self.kernel_net = torch.nn.Sequential(*kernel_net)

        # Last layer:
        self.output_linear = Linear_out(
            in_channels=hidden_channels,
            out_channels=out_channels,
            bias=bias,
        )

        # initialize the kernel function
        self.initialize(omega_0=omega_0)

        # Weight_norm
        if weight_norm:
            for (i, module) in enumerate(self.kernel_net):
                if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                    # All Conv layers are subclasses of torch.nn.Conv
                    self.kernel_net[i] = w_norm(module)

    def forward(self, x):
        out = self.kernel_net(x)
        return self.output_linear(out)

    def initialize(self, omega_0):
        net_layer = 1
        for (i, m) in enumerate(self.kernel_net.modules()):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
                if net_layer == 1:
                    w_std = 1 / m.weight.shape[1]
                    m.weight.data.uniform_(
                        -w_std, w_std
                    )  # Normally (-1, 1) / in_dim but we only use 1D inputs.
                    # Important! Bias is not defined in original SIREN implementation!
                    net_layer += 1
                else:
                    w_std = sqrt(6.0 / m.weight.shape[1]) / omega_0
                    m.weight.data.uniform_(
                        -w_std,
                        # the in_size is dim 2 in the weights of Linear and Conv layers
                        w_std,
                    )
                # TODO: Important! Bias is not defined in original SIREN implementation
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        # The final layer must be initialized differently because it is not multiplied by omega_0
        torch.nn.init.kaiming_uniform_(self.output_linear.weight, nonlinearity="linear")
        if self.output_linear.bias is not None:
            self.output_linear.bias.data.fill_(0.0)


#############################################
#       SIREN as in Sitzmann et al., 2020
##############################################
class SIREN(SIRENBase):
    """SIREN model.
    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool):
        final_activation (torch.nn.Module): Activation function.
    """

    def __init__(
        self,
        data_dim: int,
        out_channels: int,
        hidden_channels: int,
        weight_norm: bool,
        no_layers: int,
        bias: bool,
        omega_0: float,
        learn_omega_0: bool,
        **kwargs,
    ):
        # Get class of multiplied Linear Layers
        Linear_hidden = globals()[f"SIRENlayer{data_dim}d"]
        Linear_out = getattr(ckconv.nn, f"Linear{data_dim}d")

        super().__init__(
            data_dim=data_dim,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            weight_norm=weight_norm,
            no_layers=no_layers,
            bias=bias,
            omega_0=omega_0,
            learn_omega_0=learn_omega_0,
            Linear_hidden=Linear_hidden,
            Linear_out=Linear_out,
        )


class SIRENlayer1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        omega_0: float,
        learn_omega_0: bool,
        bias: bool,
    ):
        """
        Implements a Linear Layer of the form y = omega_0 * [W x + b] as in Sitzmann et al., 2020, Romero et al., 2021,
        where x is 1 dimensional.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
        )

        # omega_0
        if learn_omega_0:
            self.omega_0 = torch.nn.Parameter(torch.Tensor(1))
            with torch.no_grad():
                self.omega_0.fill_(omega_0)
        else:
            tensor_omega_0 = torch.zeros(1)
            tensor_omega_0.fill_(omega_0)
            self.register_buffer("omega_0", tensor_omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.omega_0 * torch.nn.functional.conv1d(
            x, self.weight, self.bias, stride=1, padding=0
        )

    def extra_repr(self):
        return (
            super().extra_repr() + f" omega_0={self.omega_0.item():.2f}, "
            f"learn_omega_0={self.omega_0.requires_grad}"
        )


class SIRENlayer2d(torch.nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        omega_0: float,
        learn_omega_0: bool,
        bias: bool,
    ):
        """
        Implements a Linear Layer of the form y = omega_0 * W x + b, where x is 2 dimensional
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
        )

        # omega_0
        if learn_omega_0:
            self.omega_0 = torch.nn.Parameter(torch.Tensor(1))
            with torch.no_grad():
                self.omega_0.fill_(omega_0)
        else:
            tensor_omega_0 = torch.zeros(1)
            tensor_omega_0.fill_(omega_0)
            self.register_buffer("omega_0", tensor_omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.omega_0 * torch.nn.functional.conv2d(
            x, self.weight, self.bias, stride=1, padding=0
        )

    def extra_repr(self):
        return (
            super().extra_repr() + f" omega_0={self.omega_0.item():.2f}, "
            f"learn_omega_0={self.omega_0.requires_grad}"
        )


class SIRENlayer3d(torch.nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        omega_0: float,
        learn_omega_0: bool,
        bias: bool,
    ):
        """
        Implements a Linear Layer of the form y = omega_0 * W x + b, where x is 3 dimensional
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
        )

        # omega_0
        if learn_omega_0:
            self.omega_0 = torch.nn.Parameter(torch.Tensor(1))
            with torch.no_grad():
                self.omega_0.fill_(omega_0)
        else:
            tensor_omega_0 = torch.zeros(1)
            tensor_omega_0.fill_(omega_0)
            self.register_buffer("omega_0", tensor_omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.omega_0 * torch.nn.functional.conv3d(
            x, self.weight, self.bias, stride=1, padding=0
        )

    def extra_repr(self):
        return (
            super().extra_repr() + f" omega_0={self.omega_0.item():.2f}, "
            f"learn_omega_0={self.omega_0.requires_grad}"
        )
