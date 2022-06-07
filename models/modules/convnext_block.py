# torch
import torch


class ConvNeXtBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        DepthwiseConv: torch.nn.Module,
        PointwiseConv: torch.nn.Module,
        NonlinearType: torch.nn.Module,
        NormType: torch.nn.Module,
        LinearType: torch.nn.Module,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()

        # first depthwise convolution
        self.depthwise_conv = DepthwiseConv(
            in_channels=channels,
            out_channels=channels,
            groups=1,  # should be channels for depthwise separable convolution
        )

        self.norm = NormType(channels)
        self.pointwise_conv_1 = PointwiseConv(  # TODO linear implementatie gebruiken
            in_channels=channels,
            out_channels=4 * channels,
        )
        self.act = NonlinearType()
        self.pointwise_conv_2 = PointwiseConv(
            in_channels=4 * channels,
            out_channels=channels,
        )
        self.gamma = (
            torch.nn.Parameter(
                layer_scale_init_value * torch.ones((channels)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )

        # todo: implement DropPath (stochastic network depth regularisation) https://arxiv.org/abs/1603.09382
        self.drop_path = torch.nn.Identity()

    def forward(self, x):
        # input = x x = out :-)
        input = x

        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.pointwise_conv_1(x)
        x = self.act(x)
        x = self.pointwise_conv_2(x)

        if self.gamma is not None:
            x = self.gamma[None, :, None] * x
        x = input + self.drop_path(x)
        return x
