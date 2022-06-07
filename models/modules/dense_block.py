# torch
import torch


class DenseLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_factor: float,
        ConvType: torch.nn.Module,
        NonlinearType: torch.nn.Module,
        NormType: torch.nn.Module,
        LinearType: torch.nn.Module,
    ):
        """
        TODO
        """
        super().__init__()

        # Define channels in the middle
        bottleneck_channels = int(out_channels * bottleneck_factor)

        # Norm layers
        self.norm1 = NormType(in_channels)
        self.norm2 = NormType(bottleneck_channels)

        # Conv Layers
        self.cconv1 = LinearType(
            in_channels=in_channels, out_channels=bottleneck_channels
        )  # Equal to a 1x1 convolution
        self.cconv2 = ConvType(
            in_channels=bottleneck_channels, out_channels=out_channels
        )

        # Nonlinear layer
        self.nonlinear = NonlinearType()

        self.mapping = torch.nn.Sequential(
            self.norm1,
            self.nonlinear,
            self.cconv1,
            self.norm2,
            self.nonlinear,
            self.cconv2,
        )

    def forward(self, x):
        out = self.mapping(x)
        out = torch.cat([out, x], dim=1)
        return out


class DenseBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        no_layers: int,
        bottleneck_factor: float,
        ConvType: torch.nn.Module,
        NonlinearType: torch.nn.Module,
        NormType: torch.nn.Module,
        LinearType: torch.nn.Module,
    ):
        """
        TODO
        """
        super().__init__()
        layers = []
        for layer_idx in range(no_layers):
            input_channels = in_channels + (layer_idx) * growth_rate
            output_channels = in_channels + (layer_idx + 1) * growth_rate
            layers.append(
                DenseLayer(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    bottleneck_factor=bottleneck_factor,
                    ConvType=ConvType,
                    NonlinearType=NonlinearType,
                    NormType=NormType,
                    LinearType=LinearType,
                )
            )
        self.block = torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out
