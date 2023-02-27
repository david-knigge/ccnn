import copy
import math
import random

from torch.profiler import record_function

import torch
import torch.fft
import torch.nn
import ckconv
import ckconv.nn.functional as ckconv_F


# typing
from omegaconf import OmegaConf


class PointCKConvBase(torch.nn.Module):
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
        kernel_type = kernel_cfg.type
        kernel_no_hidden = kernel_cfg.no_hidden
        kernel_no_layers = kernel_cfg.no_layers
        kernel_nonlinear = kernel_cfg.nonlinearity
        kernel_norm = kernel_cfg.norm
        kernel_omega_0 = kernel_cfg.omega_0
        kernel_bias = kernel_cfg.bias
        kernel_size = kernel_cfg.size
        kernel_chang_initialize = kernel_cfg.chang_initialize
        kernel_init_spatial_value = kernel_cfg.init_spatial_value
        kernel_num_edges = kernel_cfg.num_edges

        # We only bottleneck the convolution if we have more than 32 input channels.
        if in_channels > 32:
            kernel_bottleneck_factor = kernel_cfg.bottleneck_factor
        else:
            kernel_bottleneck_factor = -1

        # Unpack values from conv_config
        conv_bias = conv_cfg.bias
        conv_padding = conv_cfg.padding
        conv_stride = conv_cfg.stride

        # Gather kernel nonlinear and norm type
        kernel_norm = getattr(torch.nn, kernel_norm)
        kernel_nonlinear = getattr(torch.nn, kernel_nonlinear)

        if kernel_bottleneck_factor != -1:
            bottleneck_in_channels = in_channels // kernel_bottleneck_factor
        else:
            bottleneck_in_channels = in_channels

        # Define the kernel size
        if separable:
            kernel_out_channels = bottleneck_in_channels
        else:
            raise NotImplementedError("")

        # Create the kernel, using the pointconv trick this means mapping
        KernelClass = getattr(ckconv.nn.ck, kernel_type)
        self.Kernel = KernelClass(
            data_dim=data_dim,
            out_channels=kernel_out_channels,
            hidden_channels=kernel_no_hidden,
            no_layers=kernel_no_layers,
            bias=kernel_bias,
            causal=False,
            # MFN
            omega_0=kernel_omega_0,
            steerable=False,  # TODO
            init_spatial_value=kernel_init_spatial_value,
            # SIREN
            learn_omega_0=False,  # TODO
            # MLP & RFNet
            NonlinearType=kernel_nonlinear,
            NormType=kernel_norm,
            weight_norm=False,
        )

        # The kernel must have an output_linear layer. Used for chang initialization
        assert hasattr(self.Kernel, "output_linear")

        # Define convolution type
        conv_type = "conv"
        self.conv = getattr(ckconv_F, conv_type)

        # Add bias
        if conv_bias:
            if separable:
                bias_size = in_channels
            else:
                bias_size = out_channels
            self.bias = torch.nn.Parameter(torch.Tensor(bias_size))
            self.bias.data.fill_(value=0.0)
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
        self.conv_use_fft = False
        self.chang_initialize = kernel_chang_initialize
        self.separable = separable
        self.causal = False
        self.num_edges = kernel_num_edges
        self.bottleneck_factor = kernel_bottleneck_factor
        self.bottleneck = kernel_bottleneck_factor != -1
        self.bottleneck_in_channels = bottleneck_in_channels
        # 2. Non-persistent values
        self.kernel_positions = None
        self.initialized = False
        # 3. Variable placeholders
        self.register_buffer("conv_kernel", torch.zeros(1), persistent=False)

    def chang_initialization(self, num_edges):
        if not self.initialized and self.chang_initialize:
            # Initialization - Initialize the last layer of self.Kernel as in Chang et al. (2020)
            with torch.no_grad():
                if self.separable:
                    normalization_factor = num_edges
                else:
                    normalization_factor = self.in_channels * num_edges
                self.Kernel.output_linear.weight.data *= math.sqrt(
                    1.0 / normalization_factor
                )
            # Set the initialization flag to true
            self.initialized = True
