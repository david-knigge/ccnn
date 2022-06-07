# torch
import torch


# From LieConv
class Expression(torch.nn.Module):
    def __init__(self, func):
        """
        Creates a torch.nn.Module that applies the function func.
        :param func: lambda function
        """
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

    def extra_repr(self) -> str:
        return f"{self.func}"


def Sine():
    """
    out = sin(x)
    """
    return Expression(lambda x: torch.sin(x))


class GraphGELU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU()

    def forward(self, input):
        """

        :param input: PyG DataBatch object.
        """
        input.x = self.gelu(input.x)
        return input
