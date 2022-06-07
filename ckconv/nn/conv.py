import torch
import math

from omegaconf import OmegaConf

import ckconv.nn.functional as ckconv_F
import ckconv


class ConvBase(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        data_dim: int,
        kernel_cfg: OmegaConf,
        conv_cfg: OmegaConf,
        separable: bool,
        **kwargs,
    ):
        super().__init__()

        # Unpack values from kernel_config
        kernel_size = kernel_cfg.size

        # Unpack values from conv_config
        conv_use_fft = conv_cfg.use_fft
        conv_bias = conv_cfg.bias
        conv_padding = conv_cfg.padding
        conv_stride = conv_cfg.stride

        # Define convolution type
        conv_type = f"conv{data_dim}d"
        if conv_use_fft:
            conv_type = "fft" + conv_type
        if data_dim == 1:
            conv_type = "causal_" + conv_type
        self.conv = getattr(ckconv_F, conv_type)

        # Can't dynamically calculate kernel size, "full" and "same" won't work. #TODO fix this, but how?
        assert type(kernel_size) == int

        # Instantiate and initialize the weights.
        if separable:
            self.weight = torch.nn.Parameter(
                torch.empty((1, in_channels, *(kernel_size,) * data_dim))
            )
        else:
            self.weight = torch.nn.Parameter(
                torch.empty((out_channels, in_channels, *(kernel_size,) * data_dim))
            )

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if conv_bias:
            if separable:
                bias_size = in_channels
            else:
                bias_size = out_channels
            self.bias = torch.nn.Parameter(torch.Tensor(bias_size))

            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

        # Save arguments in self
        # ---------------------
        # 1. Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = conv_padding
        self.stride = conv_stride
        self.data_dim = data_dim
        self.kernel_size = kernel_size
        self.conv_use_fft = conv_use_fft
        self.separable = separable
        # 2. Variable placeholders
        self.register_buffer("train_length", torch.zeros(1).int(), persistent=True)

    def get_kernel(self, x):
        """
        Handles the vector or relative positions which is given to KernelNet.
        """
        if self.kernel_positions is None:
            if self.train_length[0] == 0:
                # Decide the extend of the rel_positions vector
                if self.kernel_size == "full":
                    self.train_length[0] = (2 * x.shape[-1]) - 1
                elif self.kernel_size == "same":
                    self.train_length[0] = x.shape[-1]
                elif int(self.kernel_size) % 2 == 1:
                    # Odd number
                    self.train_length[0] = int(self.kernel_size)
                elif int(self.kernel_size) % 2 == 0:
                    # Even number todo check whether this causes any problems
                    self.train_length[0] = int(self.kernel_size)
                else:
                    raise ValueError(
                        f"The size of the kernel must be either 'full', 'same' or an odd number"
                        f" in string format. Current: {self.kernel_size}"
                    )
        return self.kernel_positions


class Conv(ConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        data_dim: int,
        kernel_cfg: OmegaConf,
        conv_cfg: OmegaConf,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            data_dim=data_dim,
            kernel_cfg=kernel_cfg,
            conv_cfg=conv_cfg,
            separable=False,
        )

    def forward(self, x):
        return self.conv(x, self.weight, self.bias, separable=False)


class SeparableConv(ConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        data_dim: int,
        kernel_cfg: OmegaConf,
        conv_cfg: OmegaConf,
        **kwargs,
    ):
        """
        Continuous Kernel Convolution.
        :param in_channels: Number of channels in the input signal
        :param out_channels: Number of channels produced by the convolution
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            data_dim=data_dim,
            kernel_cfg=kernel_cfg,
            conv_cfg=conv_cfg,
            separable=True,
        )
        # Create the point-wise convolution
        ChannelMixerClass = getattr(ckconv.nn, f"Linear{data_dim}d")
        self.channel_mixer = ChannelMixerClass(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        # initialize
        torch.nn.init.kaiming_uniform_(self.channel_mixer.weight, nonlinearity="linear")
        torch.nn.init._no_grad_fill_(self.channel_mixer.bias, 0.0)

    def forward(self, x):
        out_unmixed = self.conv(x, self.weight, self.bias, separable=True)
        out = self.channel_mixer(out_unmixed)
        return out
