from hydra import utils
import glob
import os
import os.path as osp
import shutil

import torch
import pytorch_lightning as pl

import torch_geometric.transforms as pyg_transforms
from torch_geometric.data import Dataset, download_url, extract_zip
from torch_geometric.io import read_off, write_off
from typing import List, Optional, Union

import torch.utils.data

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.separate import separate

from .utils import (
    RenameAttribute,
    AddPosToFeat,
    GaussianRandomTranslate,
    RotateZAxisPosAndNorm,
    PreprocessDistances,
    Voxelize
)


class Collater:
    """ We use a custom collator to reshape batches, keeping in mind that we enforce all samples to contain the same
    number of points.
    """
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        sample = batch[0]

        batch = Batch.from_data_list(batch)

        # Reshape features.
        batch.x = batch.x.reshape(
            batch._num_graphs,
            *sample.x.shape,  # Needs to be consistent over all samples.
        )

        # Reshape pos vectors.
        batch.pos = batch.pos.reshape(batch._num_graphs, *sample.pos.shape)

        # Reshape rel_pos vectors.
        batch.rel_pos = batch.rel_pos.reshape(batch._num_graphs, *sample.rel_pos.shape)

        # Reshape sorted indices.
        batch.sorted_indices = batch.sorted_indices.reshape(batch._num_graphs, *sample.sorted_indices.shape)

        return batch


class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Union[Dataset, List[BaseData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(follow_batch, exclude_keys),
            **kwargs,
        )


class ModelNet(Dataset):
    r"""The ModelNet10/40 datasets from the `"3D ShapeNets: A Deep
    Representation for Volumetric Shapes"
    <https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf>`_ paper,
    containing CAD models of 10 and 40 categories, respectively.

    .. note::

        Data objects hold mesh faces instead of edge indices.
        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string, optional): The name of the dataset (:obj:`"10"` for
            ModelNet10, :obj:`"40"` for ModelNet40). (default: :obj:`"10"`)
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    urls = {
        "10": "http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        "40": "http://modelnet.cs.princeton.edu/ModelNet40.zip",
    }

    def __init__(
        self,
        root,
        name="10",
        train=True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        resampling_factor=1
    ):
        assert name in ["10", "40"]
        self.name = name
        self._resampling_factor = resampling_factor

        if train:
            self.split = "train"
        else:
            self.split = "test"

        super().__init__(root, transform, pre_transform, pre_filter)

        self._len = len(self.processed_paths)
        self._all_processed_paths = self.processed_paths


    @property
    def raw_file_names(self):
        return [
            "bathtub",
            "bed",
            "chair",
            "desk",
            "dresser",
            "monitor",
            "night_stand",
            "sofa",
            "table",
            "toilet",
        ]

    @property
    def processed_file_names(self):
        categories = glob.glob(osp.join(self.raw_dir, "*", ""))
        categories = sorted([x.split(os.sep)[-2] for x in categories])

        all_processed_paths = []
        # Construct set of processed filenames.
        for category in categories:

            # Construct set of raw filenames.
            folder = osp.join(self.processed_dir, category, self.split)
            processed_paths = glob.glob(f"{folder}/*")

            # Concatenate to list of all processed paths.
            all_processed_paths += processed_paths
        return all_processed_paths

    def download(self):
        path = download_url(self.urls[self.name], self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        folder = osp.join(self.root, f"ModelNet{self.name}")
        shutil.rmtree(self.raw_dir)
        os.rename(folder, self.raw_dir)

        # Delete osx metadata generated during compression of ModelNet10
        metadata_folder = osp.join(self.root, "__MACOSX")
        if osp.exists(metadata_folder):
            shutil.rmtree(metadata_folder)

    def process(self):
        self.process_set("train")
        self.process_set("test")

    def process_set(self, dataset):
        categories = glob.glob(osp.join(self.raw_dir, "*", ""))
        categories = sorted([x.split(os.sep)[-2] for x in categories])

        # Augment the training set by resampling mesh instances multiple times.
        if dataset == 'train' and self._resampling_factor > 1:
            resampling_number = self._resampling_factor
        else:
            resampling_number = 1

        for target, category in enumerate(categories):
            # Construct set of raw filenames.
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob(f"{folder}/{category}_*.off")

            # Obtain filenames.
            fns = [fn.split("/")[-1] for fn in paths]

            # Construct set of processed filenames.
            processed_folder = osp.join(self.processed_dir, category, dataset)

            # Create processed directory.
            os.makedirs(processed_folder, exist_ok=True)

            # Create set of processed paths.
            processed_paths = [f"{processed_folder}/{fn}" for fn in fns]

            for idx, path in enumerate(paths):

                # Augment the training set by resampling mesh instances multiple times.
                for sample_number in range(resampling_number):

                    data = read_off(path)
                    data.y = torch.tensor([target])

                    # Apply filter and pre_transform.
                    if self.pre_filter is not None:
                        data = self.pre_filter(data)

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    torch.save(data, processed_paths[idx] + f'.{sample_number}')

    def len(self):
        return self._len

    def get(self, idx: int):
        # Get data sample.
        data = torch.load(self._all_processed_paths[idx])
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.name}({len(self)})"


class ModelNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        test_batch_size,
        num_workers,
        pin_memory,
        augment,
        modelnet,
        **kwargs,
    ):
        super().__init__()

        # Get modelnet parameters.
        modelnet_name = modelnet.modelnet_name
        resampling_factor = modelnet.resampling_factor
        num_nodes = modelnet.num_nodes
        voxelize = modelnet.voxelize
        voxel_scale = modelnet.voxel_scale

        # Whether or not to voxelize the dataset.
        self.voxelize = voxelize
        self.voxel_scale = voxel_scale

        # Either 10, 40
        self.name = str(modelnet_name)

        # Save parameters to self
        if voxelize:
            self.data_dir = utils.get_original_cwd() + data_dir + "/ModelNet" + self.name + "_voxels"
        else:
            self.data_dir = utils.get_original_cwd() + data_dir + "/ModelNet" + self.name

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Determine sizes of dataset
        self.input_channels = 6  # Surface normals as input features + pos features.
        if self.name == '10':
            self.output_channels = 10  # Number of classes.
        elif self.name == '40':
            self.output_channels = 40  # Number of classes.

        # Determine data_dim & data_type
        self.data_dim = 3  # 3d pointcloud
        self.data_type = "pointcloud"

        # Create transform
        if not self.voxelize:
            pre_transform = [
                # Setup according to pointconv paper.
                pyg_transforms.SamplePoints(
                    num=num_nodes, remove_faces=True, include_normals=True
                ),
                pyg_transforms.NormalizeScale(),
                RenameAttribute(current="normal", new="x"),
                AddPosToFeat(),
                PreprocessDistances()
            ]

        # Voxelization needs to happen before processing the distances.
        else:
            pre_transform = [
                # Sample 4x the amount of points in the case of voxelization for a better result discretization.
                pyg_transforms.SamplePoints(
                    num=num_nodes * 4, remove_faces=True, include_normals=False
                ),
                pyg_transforms.NormalizeScale(),
                AddPosToFeat(),
                # Voxelize the sampled points.
                Voxelize(scale=voxel_scale, num_voxels=num_nodes),
                PreprocessDistances()
            ]

        val_test_transform = train_transform = []
        if augment:
            train_transform = val_test_transform + [
                GaussianRandomTranslate(mean=0, std=0.02),
            ]
        self.pre_transform = pyg_transforms.Compose(pre_transform)

        self.resampling_factor = resampling_factor

        self.train_transform = pyg_transforms.Compose(train_transform)
        self.val_test_transform = pyg_transforms.Compose(val_test_transform)

    def prepare_data(self):
        # download data, train then test
        ModelNet(
            root=self.data_dir,
            name=self.name,
            train=True,
            transform=self.train_transform,
            pre_transform=self.pre_transform,
            resampling_factor=self.resampling_factor
        )
        ModelNet(
            root=self.data_dir,
            name=self.name,
            train=False,
            transform=self.val_test_transform,
            pre_transform=self.pre_transform,
            resampling_factor=self.resampling_factor
        )

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == "fit" or stage is None:
            # Pointconv uses the full training dataset.
            self.train_dataset = ModelNet(
                root=self.data_dir,
                name=self.name,
                train=True,
                pre_transform=self.pre_transform,
                transform=self.train_transform,
                resampling_factor=self.resampling_factor
            )

            # Pointconv uses no validation dataset, so we use test as test and val.
            self.val_dataset = ModelNet(
                root=self.data_dir,
                name=self.name,
                train=False,
                pre_transform=self.pre_transform,
                transform=self.val_test_transform,
                resampling_factor=self.resampling_factor
            )
        if stage == "test" or stage is None:
            self.test_dataset = ModelNet(
                root=self.data_dir,
                name=self.name,
                train=False,
                pre_transform=self.pre_transform,
                transform=self.val_test_transform,
                resampling_factor=self.resampling_factor
            )

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return test_dataloader


# To support direct addition of batch features (useful for defining networks with residuals), we define batch addition
# as summing over features.
def databatch_add(batch_1, batch_2):
    batch_1.x = batch_1.x + batch_2.x
    return batch_1


setattr(Batch, "__add__", databatch_add)
setattr(Batch, "__radd__", databatch_add)
