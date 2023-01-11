
import torch
import torch.fft
import torch.nn


from .pointckconv import PointCKConvBase

# typing
from omegaconf import OmegaConf


class PointFlexConvBase(PointCKConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        data_dim: int,
        kernel_cfg: OmegaConf,
        conv_cfg: OmegaConf,
        mask_cfg: OmegaConf,
        separable: bool,
        **kwargs,
    ):
        # Unpack mask_config values:
        mask_type = mask_cfg.type
        mask_init_value = mask_cfg.init_value
        mask_learn_mean = mask_cfg.learn_mean
        mask_temperature = mask_cfg.temperature
        mask_threshold = mask_cfg.threshold

        if mask_type == "gaussian":
            init_spatial_value = mask_init_value * 1.667
        else:
            raise NotImplementedError(f"Mask of type '{mask_type}' not implemented.")

        # Overwrite init_spatial value
        kernel_cfg.init_spatial_value = init_spatial_value

        # call super class
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            data_dim=data_dim,
            kernel_cfg=kernel_cfg,
            conv_cfg=conv_cfg,
            separable=separable,
        )

        # Define mask constructor
        self.mask_constructor = globals()[f"{mask_type}_mask_{self.data_dim}d"]

        # Define learnable parameters of the mask, maps from data dimensionality to parameter
        mask_width_param = {
            "gaussian": {
                3: torch.Tensor([mask_init_value])
            },
        }[mask_type][self.data_dim]
        self.mask_width_param = torch.nn.Parameter(mask_width_param)

        mask_mean_param = {
            "gaussian": {
                3: torch.Tensor([0.0]),
            },
        }[mask_type][self.data_dim]
        if mask_learn_mean:
            self.mask_mean_param = torch.nn.Parameter(mask_mean_param)
        else:
            self.register_buffer("mask_mean_param", mask_mean_param)

        self.root_function = gaussian_max_abs_root

        self.mask_temperature = mask_temperature

        # Store mask threshold.
        mask_threshold = mask_threshold * torch.ones(1)
        self.register_buffer("mask_threshold", mask_threshold, persistent=True)

        # Save values in self, dynamic cropping isn't supported for pointcloud data.
        self.dynamic_cropping = False

    def sample_kernel_points(self, x, pos):
        # This now happens for each kernel separately.
        pass

    def sample_and_crop_kernel_points(self, rel_pos, x, sorted_indices):
        """ For every point in the input, sample the 'self.num_edges' nearest points and their corresponding features.
        """

        # 1. WITH FIXED NUMBER OF SAMPLES PER NEIGHBOURHOOD
        edge_indices = sorted_indices[:, :, :self.num_edges]

        # Obtain the relative positions for the N nearest points.
        rel_edge_pos = torch.gather(
            input=rel_pos,
            dim=2,
            index=edge_indices.unsqueeze(3).expand(
                    *edge_indices.shape,
                    rel_pos.shape[-1]
                )
        )

        # Obtain the values at these positions. This results in a
        sampled_x = torch.gather(
            input=x.unsqueeze(2).expand(
                x.shape[0],
                x.shape[1],
                x.shape[1],
                x.shape[2]),
            dim=2,
            index=edge_indices.unsqueeze(3).expand(
                *edge_indices.shape,
                x.shape[-1]
            )
        )

        return rel_edge_pos, sampled_x, edge_indices


class PointFlexConv(PointFlexConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        data_dim: int,
        kernel_cfg: OmegaConf,
        conv_cfg: OmegaConf,
        mask_cfg: OmegaConf,
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
            mask_cfg=mask_cfg,
            separable=False,
        )

    def forward(self, **kwargs):
        raise NotImplementedError("This implementation would be very memory consuming.")


class SeparablePointFlexConv(PointFlexConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        data_dim: int,
        kernel_cfg: OmegaConf,
        conv_cfg: OmegaConf,
        mask_cfg: OmegaConf,
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
            mask_cfg=mask_cfg,
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

        # Get data statistics.
        x = data.x
        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        num_edges = self.num_edges

        # Obtain data on relative kernel positions.
        rel_pos = data.rel_pos
        sorted_indices = data.sorted_indices

        # Sample the kernel points.
        kernel_pos, sampled_x, kernel_pos_idx = self.sample_and_crop_kernel_points(rel_pos, x, sorted_indices)

        # 1. Construct kernel mask, crop pos vector accordingly.
        mask = self.mask_constructor(
            kernel_pos.flatten(0, -2),
            self.mask_mean_param,
            self.mask_width_param,
        )

        # 2. Construct kernel. [num_edges, in_channels * out_channels]
        kernel_values = self.Kernel(kernel_pos.view(
            data.x.shape[0] * data.x.shape[1] * kernel_pos.shape[2], 3, 1, 1, 1
        )).squeeze()

        # 2.1 Apply mask to kernel.
        masked_kernel_values = kernel_values * mask.unsqueeze(1)

        # 2.2 Reshape to conv kernel.
        masked_kernel_values = masked_kernel_values.view(
            batch_size,  # number of graphs
            num_nodes,  # number of nodes
            num_edges,  # number of neighbours
            self.bottleneck_in_channels,
        )

        self.conv_kernel = masked_kernel_values

        # 3. Perform convolution TODO use pointconv trick?
        # Currently, we are able to use batch matrix multiplication as the graphs and neighbourhoods we are convolving
        # over are equal in size. How do we rewrite this to be applicable to say QM9?

        # Multiply kernel values with values at node positions, sum over all edges.
        # Results in [batch_size, num_nodes, in_channels].

        # We transform all features in x.
        if self.bottleneck:
            x = self.bottleneck_linear(x)

        # Map input to output channels, [batch_size, num_nodes, in_channels].
        data.x = self.channel_mixer(
            (torch.gather(
                input=x.unsqueeze(2).expand(
                    x.shape[0],
                    x.shape[1],
                    x.shape[1],
                    x.shape[2]),
                dim=2,
                index=kernel_pos_idx.unsqueeze(3).expand(
                    *kernel_pos_idx.shape,
                    x.shape[-1]
                )
            ) * masked_kernel_values).sum(dim=2)
        )
        return data


###############################
# Gaussian Masks / Operations #
###############################
def gaussian_mask_3d(
    rel_positions: torch.Tensor,
    mask_mean_param: torch.Tensor,
    mask_width_param: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    mask_y = gaussian_function(
        rel_positions[:, 0], mask_mean_param[0], mask_width_param[0]
    )
    mask_x = gaussian_function(
        rel_positions[:, 1], mask_mean_param[0], mask_width_param[0]
    )
    mask_z = gaussian_function(
        rel_positions[:, 2], mask_mean_param[0], mask_width_param[0]
    )

    mask = mask_y * mask_x * mask_z
    return mask


def gaussian_function(
    x: torch.Tensor,
    mean: torch.Tensor,
    sigma: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return torch.exp(-1.0 / 2 * ((1.0 / sigma) * (x - mean)) ** 2)


def gaussian_inv_thresh(
    thresh: float,
    mean: float,
    sigma: float,
    **kwargs,
):
    # Based on the threshold value, compute the value of the roots
    aux = sigma * torch.sqrt(-2.0 * torch.log(thresh))
    return mean - aux, mean + aux


def gaussian_max_abs_root(
    thresh: float,
    mean: float,
    sigma: float,
    **kwargs,
):
    return max(map(abs, gaussian_inv_thresh(thresh, mean, sigma)))

