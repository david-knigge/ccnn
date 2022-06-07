# torch
import torch


class S4Block(torch.nn.Module):
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
        prenorm: bool = True,
    ):
        """
        Instantiates the core elements of a residual block with only one convolution
        These elements are:
        (1) 1 convolutional layers
        (2) Two normalization layers
        (3) A residual connection
        (4) A dropout layer
        """
        super().__init__()

        # Conv Layers
        self.conv1 = ConvType(in_channels=in_channels, out_channels=out_channels)

        # Nonlinear layer
        self.nonlinears = torch.nn.ModuleList((NonlinearType(), NonlinearType()))

        # Norm layers
        self.prenorm = prenorm
        if prenorm:
            norm_channels = in_channels
        else:
            norm_channels = out_channels
        self.norm1 = NormType(norm_channels)

        # Linear layers
        self.linear1 = LinearType(out_channels, out_channels)
        # initialize linear layers
        torch.nn.init.kaiming_normal_(self.linear1.weight)
        if self.linear1.bias is not None:
            self.linear1.bias.data.fill_(value=0.0)

        # Dropout
        self.dp = DropoutType(dropout)

        # Shortcut
        shortcut = []
        if in_channels != out_channels:
            shortcut.append(LinearType(in_channels, out_channels))
            torch.nn.init.kaiming_normal_(shortcut[0].weight)
            if shortcut[0].bias is not None:
                shortcut[0].bias.data.fill_(value=0.0)
        self.shortcut = torch.nn.Sequential(*shortcut)

    def forward(self, x):
        shortcut = self.shortcut(x)
        # if prenorm: Norm -> Conv -> Nonlinear -> Linear -> Sum
        # else: Conv -> Nonlinear -> Linear -> Sum -> Norm
        if self.prenorm:
            x = self.norm1(x)
        out = self.nonlinears[1](
            self.linear1(self.dp(self.nonlinears[0](self.conv1(x))))
        )
        out = out + shortcut
        if not self.prenorm:
            out = self.norm1(out)
        return out
