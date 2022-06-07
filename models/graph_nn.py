import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing

# project
import ckconv

# typing
from omegaconf import OmegaConf


class GraphConv(MessagePassing):
    def __init__(
        self,
        data_dim,
        in_channels,
        out_channels,
        bias,
    ):
        super().__init__(
            aggr="add",
            flow="source_to_target",  # From neigh. node j to node i.
        )
        self.in_channels = in_channels
        self.out_channels = self.out_channels
        self.weight = torch.nn.Linear(data_dim, in_channels * out_channels)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def forward(self, x, pos, batch):
        # Sample a set of farthest points, idx contains indices corresponding to these points in pos
        # so pos[idx] is a set of points that lie farthest away from each other
        idx = pyg.nn.fps(pos, batch, ratio=self.ratio)

        # Radius gives us, for each of the farthest points in pos[idx], the points in pos that are at most self.radius away
        # from it. Row are indices in pos[idx], col are indices in pos.
        pos_i, pos_j = pyg.nn.radius(
            pos,
            pos[idx],
            self.radius,
            batch,
            batch[idx],
            max_num_neighbors=64,
        )

        # Edge_index is the set of edges from nodes in pos to nodes in pos[idx].
        edge_index = torch.stack([pos_i, pos_j], dim=0)


class SAModule(torch.nn.Module):
    def __init__(
        self,
        data_dim,
        in_channels,
        out_chanels,
        KernelType,
        ratio,
        radius,
    ):
        super().__init__()

        self.ratio = ratio
        self.radius = radius

        self.neural_net = KernelType(
            [
                data_dim + in_channels,
                64,
                64,
                out_chanels,
            ],
        )

        # Perform a PointNetConv with the neural network nn.
        self.conv = pyg.nn.PointConv(self.neural_net, add_self_loops=False)

    def forward(self, x, pos, batch):
        # Sample a set of farthest points, idx contains indices corresponding to these points in pos
        # so pos[idx] is a set of points that lie farthest away from each other
        idx = pyg.nn.fps(pos, batch, ratio=self.ratio)

        # Radius gives us, for each of the farthest points in pos[idx], the points in pos that are at most self.radius away
        # from it. Row are indices in pos[idx], col are indices in pos.
        row, col = pyg.nn.radius(
            pos, pos[idx], self.radius, batch, batch[idx], max_num_neighbors=64
        )

        # Edge_index is the set of edges from nodes in pos to nodes in pos[idx].
        edge_index = torch.stack([col, row], dim=0)

        # Optionally include node features.
        x_dst = None if x is None else x[idx]

        # Perform convolution where the output is defined over pos[idx].
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)

        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = pyg.nn.global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet_default(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        net_cfg: OmegaConf,
        kernel_cfg: OmegaConf,
        conv_cfg: OmegaConf,
        mask_cfg: OmegaConf,
    ):
        super().__init__()

        # Unpack arguments from net_config
        hidden_channels = net_cfg.no_hidden
        no_blocks = net_cfg.no_blocks
        data_dim = net_cfg.data_dim
        norm = net_cfg.norm
        dropout = net_cfg.dropout
        dropout_in = net_cfg.dropout_in
        block_type = net_cfg.block_type
        block_width_factors = net_cfg.block_width_factors
        nonlinearity = net_cfg.nonlinearity

        # Define NormType
        if norm == "BatchNorm":
            norm_name = f"BatchNorm{data_dim}d"
        else:
            norm_name = norm
        if hasattr(ckconv.nn, norm):
            lib = ckconv.nn
        else:
            lib = torch.nn
        NormType = getattr(lib, norm_name)

        # Define NonlinearType
        NonlinearType = getattr(torch.nn, nonlinearity)

        # Define LinearType
        KernelType = pyg.nn.MLP

        # 2. Create blocks
        # Create Blocks
        # -------------------------
        # 1. Create vector of width_factors:
        # If value is zero, then all values are one
        if block_width_factors[0] == 0.0:
            width_factors = (1,) * no_blocks
        else:
            width_factors = [
                (factor,) * n_blcks
                for factor, n_blcks in ckconv.utils.pairwise_iterable(
                    block_width_factors
                )
            ]
            width_factors = [
                factor for factor_tuple in width_factors for factor in factor_tuple
            ]
        if len(width_factors) != no_blocks:
            raise ValueError(
                "The size of the width_factors does not matched the number of blocks in the network."
            )
        blocks = []
        for i in range(no_blocks):
            print(f"Block {i}/{no_blocks}")

            if i == 0:
                input_ch = in_channels
                hidden_ch = int(hidden_channels * width_factors[i])
            else:
                input_ch = int(hidden_channels * width_factors[i - 1])
                hidden_ch = int(hidden_channels * width_factors[i])

            blocks.append(
                (
                    SAModule(
                        data_dim=data_dim,
                        in_channels=input_ch,
                        out_chanels=hidden_ch,
                        KernelType=KernelType,
                        ratio=1.0,
                        radius=2.0,
                    ),
                    "x, pos, batch -> x, pos, batch",
                )
            )

        self.blocks = pyg.nn.Sequential(input_args="x, pos, batch", modules=blocks)

        # Define Output Layers:
        # -------------------------
        # 1. Calculate output channels of blocks
        if block_width_factors[0] == 0.0:
            final_no_hidden = hidden_channels
        else:
            final_no_hidden = int(hidden_channels * block_width_factors[-2])
        # 2. instantiate last layer
        self.out_layer = KernelType(
            [
                final_no_hidden,
                64,
                64,
                out_channels,
            ]
        )

        # Define global pooling layer:
        self.global_pooling = GlobalSAModule(
            KernelType(
                [
                    final_no_hidden + data_dim,
                    64,
                    64,
                    final_no_hidden,
                ],
            )
        )

    def forward(self, x):
        x = (x.x, x.pos, x.batch)
        out = self.blocks(*x)
        out, out_pos, out_batch = self.global_pooling(*out)
        return self.out_layer(out)
