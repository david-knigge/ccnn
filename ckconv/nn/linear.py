import torch
import torch.nn as nn


def Linear1d(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True,
) -> torch.nn.Module:
    """
    Implements a Linear Layer in terms of a point-wise convolution.
    """
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def Linear2d(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True,
) -> torch.nn.Module:
    """
    Implements a Linear Layer in terms of a point-wise convolution.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def Linear3d(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True,
) -> torch.nn.Module:
    """
    Implements a Linear Layer in terms of a point-wise convolution.
    """
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


class GraphLinear(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

    def get_weight(self):
        return self.lin.weight

    def get_bias(self):
        return self.lin.bias

    weight = property(get_weight)
    bias = property(get_bias)

    def forward(self, input):
        """

        :param input: PyG Data Batch object.
        """
        input.x = self.lin(input.x)
        return input
