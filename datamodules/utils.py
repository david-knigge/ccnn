import torch
import os
import sklearn.model_selection
import math
import numbers
import random

from functools import partial

from torch import Tensor

from torch_geometric.transforms import BaseTransform, LinearTransformation, GridSampling
from torch_cluster import grid_cluster

import matplotlib.pyplot as plt


def pad(channel, maxlen):
    channel = torch.tensor(channel)
    out = torch.full((maxlen,), channel[-1])
    out[: channel.size(0)] = channel
    return out


def subsample(X, y, subsample_rate):
    if subsample_rate != 1:
        X = X[:, :, ::subsample_rate]
    return X, y


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + ".pt")


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith(".pt"):
            tensor_name = filename.split(".")[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors


def load_data_from_partition(data_loc, partition):
    assert partition in ["train", "val", "test"]
    # load tensors
    tensors = load_data(data_loc)
    # select partition
    name_X, name_y = f"{partition}_X", f"{partition}_y"
    X, y = tensors[name_X], tensors[name_y]
    return X, y


def normalise_data(X, y):
    train_X, _, _ = split_data(X, y)
    out = []
    for Xi, train_Xi in zip(X.unbind(dim=-1), train_X.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()  # compute statistics using only training data.
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    out = torch.stack(out, dim=-1)
    return out


def split_data(tensor, stratify):
    # 0.7/0.15/0.15 train/val/test split
    (
        train_tensor,
        testval_tensor,
        train_stratify,
        testval_stratify,
    ) = sklearn.model_selection.train_test_split(
        tensor,
        stratify,
        train_size=0.7,
        random_state=0,
        shuffle=True,
        stratify=stratify,
    )

    val_tensor, test_tensor = sklearn.model_selection.train_test_split(
        testval_tensor,
        train_size=0.5,
        random_state=1,
        shuffle=True,
        stratify=testval_stratify,
    )
    return train_tensor, val_tensor, test_tensor


class FullyConnectedGraph(BaseTransform):
    r"""Creates edges based on node positions :obj:`pos` to all points within a
    given distance.

    Args:
        r (float): The distance.
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`.
            This flag is only needed for CUDA tensors. (default: :obj:`32`)
        flow (string, optional): The flow direction when using in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    """

    def __init__(self, loop: bool = True):
        self.loop = loop

    def __call__(self, data):
        data.edge_attr = None

        # Create a list containing all nodes.
        nodelist = torch.arange(start=0, end=data.pos.shape[0], step=1).long()

        # Fully connected edge list of shape [num_edges, 2].
        edge_index = torch.cartesian_prod(nodelist, nodelist)

        # Transpose to obtain correct shape.
        data.edge_index = edge_index.T
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r={self.r})"


class RenameAttribute(BaseTransform):
    """TODO: Implement support for retaining existing attribute with same name as 'new'."""

    def __init__(
        self,
        current: str,
        new: str,
    ):
        self.current = current
        self.new = new

    def __call__(self, data):
        # Replace attribute name.
        if hasattr(data, self.current):
            setattr(data, self.new, getattr(data, self.current))
            setattr(data, self.current, None)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class NormalizeEdgeAttr(BaseTransform):
    """TODO: Implement support for retaining existing attribute with same name as 'new'."""

    def __init__(
        self,
        min: float,
        max: float,
    ):
        self.min = min
        self.max = max

    def __call__(self, data):
        # Normalize the edge features.
        if "edge_attr" in data:
            max_value = data.edge_attr.max()
            min_value = data.edge_attr.min()

            data.edge_attr = (data.edge_attr - min_value) / (max_value - min_value)

            # Calculate range to which features are normalized.
            absdiff = self.max - self.min

            # Scale to this range, subtract
            data.edge_attr = data.edge_attr * absdiff + self.min

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AddPosToFeat(BaseTransform):
    def __init__(
        self,
    ):
        pass

    def __call__(self, data):
        if "pos" in data:
            if "x" in data:
                # Final dimension is channel dim.
                data.x = torch.cat((data.x, data.pos), dim=-1)
            else:
                data.x = data.pos
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class GaussianRandomTranslate(BaseTransform):
    def __init__(self, mean, std, clip=0.05):
        self.mean = mean
        self.std = std
        self.clip = clip

    def __call__(self, data):
        # Add Gaussian random noise.
        noise = torch.clip(
            torch.normal(self.mean, self.std, size=data.pos.shape),
            min=-self.clip,
            max=self.clip,
        )
        data.pos += noise
        if "x" in data:
            data.x[:, 3:] = data.pos
        return data


class RotateZAxisPosAndNorm(BaseTransform):
    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees

    def __call__(self, data):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        rot = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]).to(
            data.x.device, data.x.dtype
        )
        data.pos = data.pos @ rot.T
        if "x" in data:
            data.x[:, 3:] = data.pos
            data.x[:, :3] = data.x[:, :3] @ rot.T
        return data


class PreprocessDistances(BaseTransform):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        # Store pairwise distances.
        data.rel_pos = data.pos.unsqueeze(1) - data.pos.unsqueeze(0)

        # Use euclidean distance to sort indices.
        cdist = torch.cdist(data.pos, data.pos)

        # Sort euclidean pairwise distances.
        sorted_cdist, sorted_indices = torch.sort(cdist, dim=1)

        # Store indices.
        data.sorted_indices = sorted_indices

        return data


class Voxelize(BaseTransform):

    def __init__(self, scale, num_voxels):
        self.scale = scale
        self.num_voxels = num_voxels

        self.start = -1.0 - self.scale
        self.end = 1.0 + self.scale

        self.grid_sampling = GridSampling(self.scale, self.start, self.end)

    def __call__(self, data):

        # Grid sample the input data.
        data = self.grid_sampling(data)

        indices = np.arange(data.pos.shape[0])
        if data.pos.shape[0] < self.num_voxels:
            print("Number of nonzero voxels is smaller than number of nodes to be sampled, sampling with replacement.")
            selected_indices = np.random.choice(indices, size=self.num_voxels, replace=True)
        else:
            # Randomly select subset of voxels
            np.random.shuffle(indices)
            selected_indices = indices[:self.num_voxels]

        data.pos = data.pos[selected_indices, :]

        # Round to nearest gridpoint.
        def round_base(x, base):
            return base * round(float(x) / base)

        rounder = partial(round_base, base=self.scale)
        data.pos.apply_(rounder)

        # Overwrite the position data in features.
        data.x = data.pos

        return data
