import torch

import typing

import numpy as np
from plotly import graph_objects as go


def visualize_tensor_1d(tensor: torch.Tensor, limit=10) -> typing.List[go.Figure]:

    figs = []
    spatial_shape_1d = tensor.shape[-1]
    num_in_channels = tensor.shape[1]
    flat_tensor = torch.flatten(tensor.detach().cpu(), start_dim=1, end_dim=-2)

    i = 0

    for t in flat_tensor:

        fig = go.Figure(
            data=go.Heatmap(
                x=np.linspace(0, spatial_shape_1d, spatial_shape_1d),
                y=np.linspace(0, num_in_channels, num_in_channels),
                z=t,
                colorscale="Viridis",
            )
        )

        figs.append(fig)
        i += 1
        if i > limit:
            return figs
    return figs


def visualize_tensor_2d(tensor: torch.Tensor, limit=10) -> typing.List[go.Figure]:

    figs = []
    spatial_1 = tensor.shape[-1]
    spatial_2 = tensor.shape[-2]
    num_in_channels = tensor.shape[1]
    flat_tensor = torch.flatten(tensor.detach().cpu(), start_dim=1, end_dim=-3)

    i = 0

    for out_channel in flat_tensor:

        for in_channel in out_channel:

            fig = go.Figure(
                data=go.Heatmap(
                    x=np.linspace(0, spatial_1, spatial_1),
                    y=np.linspace(0, spatial_2, spatial_2),
                    z=in_channel,
                    colorscale="Viridis",
                )
            )

            figs.append(fig)
            i += 1
            if i > limit:
                return figs

    return figs
