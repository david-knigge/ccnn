import torch
import ckconv
from torch.nn.utils import weight_norm as w_norm

# Based on https://github.com/lucidrains/siren-pytorch
from math import sqrt


class MLPBase(torch.nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        no_layers: int,
        LinearType: torch.nn.Module,
        NonlinearType: torch.nn.Module,
        NormType: torch.nn.Module,
        weight_norm: bool,
        bias: bool,
    ):
        super().__init__()

        # Construct the network
        # ---------------------
        # Hidden layers:
        hidden_layers = []
        for _ in range(no_layers - 2):
            hidden_layers.extend(
                [
                    LinearType(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        bias=bias,
                    ),
                    NormType(hidden_channels),
                    NonlinearType(),
                ]
            )
        self.hidden_layers = torch.nn.Sequential(*hidden_layers)

        # Last layer:
        self.output_linear = LinearType(hidden_channels, out_channels, bias=bias)

        # initialize the kernel function
        self.initialize(
            NonlinearType,
        )

        # Weight_norm
        if weight_norm:
            # Hidden layers
            for (i, module) in enumerate(self.hidden_layers):
                if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                    # All Conv layers are subclasses of torch.nn.Conv
                    self.kernel_net[i] = w_norm(module)
            # Output layer
            self.output_linear = w_norm(self.output_linear)

    def forward(self, x):
        out = self.input_layers(x)
        out = self.hidden_layers(out)
        return self.output_linear(out)

    def initialize(
        self,
        NonlinearType: torch.nn.Module,
    ):
        # Define the gain
        if NonlinearType == torch.nn.ReLU:
            nonlin = "relu"
        elif NonlinearType == torch.nn.LeakyReLU:
            nonlin = "leaky_relu"
        else:
            nonlin = "linear"

        # Initialize hidden layers
        for (i, m) in enumerate(self.hidden_layers.modules()):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlin)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
        # Initialize output layer
        torch.nn.init.kaiming_uniform_(
            self.output_linear.weight, nonlinearity="linear"
        )  # TODO: Define based on the nonlin used in the main network.
        if self.output_linear.bias is not None:
            torch.nn.init.constant_(self.output_linear.bias, 0.0)


class MLP(MLPBase):
    def __init__(
        self,
        data_dim: int,
        out_channels: int,
        hidden_channels: int,
        no_layers: int,
        NonlinearType: torch.nn.Module,
        NormType: torch.nn.Module,
        weight_norm: bool,
        bias: bool,
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

        # Construct 1st layers (Sequence of modules, e.g., input embeddings)
        input_layers = [
            LinearType(
                in_channels=data_dim,
                out_channels=hidden_channels,
                bias=bias,
            ),
            NormType(hidden_channels),
            NonlinearType(),
        ]
        self.input_layers = torch.nn.Sequential(*input_layers)

        # Initialize the input layers
        self.initialize_input_layers(NonlinearType)

    def initialize_input_layers(
        self,
        NonlinearType: torch.nn.Module,
    ):
        # Define the gain
        if NonlinearType == torch.nn.ReLU:
            nonlin = "relu"
        elif NonlinearType == torch.nn.LeakyReLU:
            nonlin = "leaky_relu"
        else:
            nonlin = "linear"

        # Initialize hidden layers
        for (i, m) in enumerate(self.input_layers.modules()):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlin)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
