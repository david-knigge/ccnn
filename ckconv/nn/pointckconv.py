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

    def sample_kernel_points(self, x, pos):
        """

        :param x: [batch_size, num_nodes, num_channels]
        :param pos: [batch_size, num_nodes, data_dim]
        :param num_edges: int, number of edges to sample kernel values for. Analogous to kernel size.
        """
        no_edges = self.num_edges
        bs, no_nodes, data_dim = pos.shape

        # Number of sampled edges needs to be smaller than total number of nodes.
        assert no_edges < no_nodes

        # # Randomly samply `num_edges` nodes to serve as edge nodes. # [num_edges]
        # perm = torch.randperm(num_nodes)
        # edge_indices = perm[:num_edges]

        edge_indices = torch.stack(
            [
                torch.tensor(random.sample(range(no_nodes), no_edges), device=x.device)
                for _ in range(no_nodes)
            ],
            dim=0,
        )
        rel_edge_pos = pos.unsqueeze(2) - pos[:, edge_indices, :]

        # # Obtain positions for the nodes that serve as edge. # [batch, num_edges, data_dim]
        # edge_pos = pos[:, edge_indices, :]
        #
        # # # Repeat for number of nodes. # [batch, num_nodes, num_edges, data_dim]
        # # edge_pos = edge_pos.unsqueeze(1).repeat(1, num_nodes, 1, 1)
        # #
        # # # Repeat positions of center nodes. # [batch, num_nodes, num_edges, data_dim]
        # # center_pos = pos.unsqueeze(2).repeat(1, 1, num_edges, 1)
        #
        # # Calculate relative cartesian coordinates for each of the edges. # [batch, num_nodes, num_edges, data_dim]
        # rel_edge_pos = pos.unsqueeze(2) - edge_pos.unsqueeze(1)

        # # Calculate max and min vals
        # max_value = rel_edge_pos.amax(dim=-2).unsqueeze(2)
        # min_value = rel_edge_pos.amin(dim=-2).unsqueeze(2)
        # # Scale rel_edge_pos to [0, 1]
        # rel_edge_pos = (rel_edge_pos - min_value) / (max_value - min_value)
        # # Scale rel_edge_pos to interval [-1, 1]
        # rel_edge_pos = rel_edge_pos * 2 - 1

        return rel_edge_pos, edge_indices


class PointCKConv(PointCKConvBase):
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
            separable=False,
        )

        # Todo implement PointConv trick. Do convolution on first to last layer of KernelNet.
        # This means linear mapping from kernel_net_hidden_size to in_channels*out_channels here.

    def forward(self, **kwargs):
        raise NotImplementedError("This implementation would be very memory consuming.")


class SeparablePointCKConv(PointCKConvBase):
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
        # Create bottleneck.
        if self.bottleneck:
            self.bottleneck_linear = torch.nn.Linear(
                self.in_channels, self.bottleneck_in_channels, bias=conv_cfg.bias
            )
            with torch.no_grad():
                torch.nn.init.kaiming_uniform_(
                    self.bottleneck_linear.weight, nonlinearity="linear"
                )
                torch.nn.init._no_grad_fill_(self.bottleneck_linear.bias, 0.0)
        else:
            self.bottleneck_linear = None

        # Create the point-wise convolution.
        self.channel_mixer = torch.nn.Linear(
            self.bottleneck_in_channels, self.out_channels, bias=conv_cfg.bias
        )
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(
                self.channel_mixer.weight, nonlinearity="linear"
            )
            torch.nn.init._no_grad_fill_(self.channel_mixer.bias, 0.0)

    def forward(self, data):
        """
        :param x: Node features (like surface normals).
        :param pos: Node positions (not used) [num_nodes, num_dimensions].
        :param edge_index: Edges [2, num_edges].
        :param edge_attr: Edge features, distances are stored here [num_edges, num_edge_features (should be 3?)].
        """

        # Perform chang initialization if not done yet.
        self.chang_initialization(self.num_edges)

        x = data.x
        kernel_pos = data.kernel_pos
        kernel_pos_idx = data.kernel_pos_idx
        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        num_edges = self.num_edges

        # 2. Construct kernel. Should take pointconv trick into account. [num_edges, in_channels * out_channels]
        kernel_values = self.Kernel(kernel_pos).view(
            batch_size,  # number of graphs
            num_nodes,  # number of nodes
            num_edges,  # number of neighbours
            self.bottleneck_in_channels,
        )

        # 3. Perform convolution TODO use pointconv trick?
        # Currently, we are able to use batch matrix multiplication as the graphs and neighbourhoods we are convolving
        # over are equal in size. How do we rewrite this to be applicable to say QM9?

        # Multiply kernel values with values at node positions, sum over all edges.
        # Results in [batch_size, num_nodes, in_channels].
        # x = (x[:, kernel_pos_idx, :].unsqueeze(1) * kernel_values).sum(dim=2)
        # TODO rewrite efficiently.
        # TODO: We need the proper kernel_pos_idx stuff. Probably we need different nbhs per point.

        # We transform all features in x.
        if self.bottleneck:
            x = self.bottleneck_linear(x)

        # Map input to output channels, [batch_size, num_nodes, in_channels].
        data.x = self.channel_mixer(
            (x[:, kernel_pos_idx, :] * kernel_values).sum(dim=2)
        )
        return data
