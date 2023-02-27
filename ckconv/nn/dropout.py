import torch
from torch import Tensor


class GraphDropout(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dp = torch.nn.Dropout(p=p)

    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: PyG batch object.
        """
        input.x = self.dp(input.x)
        return input


class GraphDropout2d(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dp = torch.nn.Dropout2d(p=p)

    def forward(self, input: Tensor) -> Tensor:
        """

        :param input: PyG batch object.
        """

        input.x = self.dp(input.x)
        return input
