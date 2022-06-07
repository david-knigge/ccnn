# torch
import torch


class ResidualBlockBase(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ConvType: torch.nn.Module,
        NonlinearType: torch.nn.Module,
        NormType: torch.nn.Module,
        LinearType: torch.nn.Module,
        DropoutType: torch.nn.Module,
        dropout: float,
        **kwargs,
    ):
        """
        Instantiates the core elements of a residual block but does not implement the forward function.
        These elements are:
        (1) Two convolutional layers
        (2) Two normalization layers
        (3) A residual connection
        (4) A dropout layer
        """
        super().__init__()

        # Conv Layers
        self.conv1 = ConvType(in_channels=in_channels, out_channels=out_channels)
        self.conv2 = ConvType(in_channels=out_channels, out_channels=out_channels)

        # Nonlinear layer
        self.nonlinearities = torch.nn.ModuleList(
            (NonlinearType(), NonlinearType(), NonlinearType())
        )

        # Norm layers
        self.norm1 = NormType(out_channels)
        self.norm2 = NormType(out_channels)

        # Dropout
        self.dp = DropoutType(dropout)

        # Shortcut
        shortcut = []
        if in_channels != out_channels:
            shortcut.append(LinearType(in_channels, out_channels))
            torch.nn.init.kaiming_normal_(shortcut[0].weight)
            if shortcut[0].bias is not None:
                shortcut[0].bias.data.fill_(value=0.0)
            print("shortcut used")
        self.shortcut = torch.nn.Sequential(*shortcut)

    def forward(self, x):
        raise NotImplementedError()


class TCNBlock(ResidualBlockBase):
    """
    Creates a Residual Block alike to TCNs ( Bai et al., 2017 ). Differently, we make the nonlinearities and
    norm classes flexible.

    input
     | ---------------|
     Conv             |
     Norm             |
     NonLinearity     |
     DropOut          |
     |                |
     Conv             |
     Norm             |
     NonLinearity     |
     DropOut          |
     + <--------------|
     |
     NonLinearity
     |
     output
    """

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.dp(self.nonlinearities[0](self.norm1(self.conv1(x))))
        out = self.dp(self.nonlinearities[1](self.norm2(self.conv2(out)))) + shortcut
        out = self.nonlinearities[2](out)
        return out


class ResNetBlock(ResidualBlockBase):
    """
    Creates a Residual Block as in the original ResNet paper (He et al., 2016)
    input
     | ---------------|
     Conv             |
     Norm             |
     NonLinearity     |
     |                |
     DropOut          |
     |                |
     Conv             |
     Norm             |
     |                |
     + <--------------|
     |
     NonLinearity
     |
     output
    """

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.dp(self.nonlinearities[0](self.norm1(self.conv1(x))))
        out = self.norm2(self.conv2(out)) + shortcut
        out = self.nonlinearities[1](out)
        return out


class PreActResNetBlock(ResidualBlockBase):
    """
    Creates a Residual Block as in the original ResNet paper (He et al., 2016)
    input
     | ---------------|
     Norm             |
     NonLinearity     |
     Conv             |
     |                |
     DropOut          |
     |                |
     Norm             |
     NonLinearity     |
     Conv             |
     |                |
     + <--------------|
     |
     output
    """

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.dp(self.conv1(self.nonlinearities[0](self.norm1(x))))
        out = self.conv2(self.nonlinearities[1](self.norm2(out))) + shortcut
        return out
