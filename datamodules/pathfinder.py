import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl

from PIL import Image
from pathlib import Path

# config
from hydra import utils

# typing
from typing import Optional, Callable, Tuple, Dict, List, cast


# There's an empty file in the dataset
PATHFINDER_BLACKLIST = {"pathfinder32/curv_baseline/imgs/0/sample_172.png"}


def pil_loader_grayscale(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        return Image.open(f).convert("L")


class PathFinderDataset(datasets.ImageFolder):
    """Path Finder dataset."""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super().__init__(
            root,
            loader=pil_loader_grayscale,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Override this so it doesn't call the parent's method"""
        return [], {}

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).
        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.
        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.
        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.
        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        # We ignore class_to_idx
        directory = Path(directory).expanduser()

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError(
                "Both extensions and is_valid_file cannot be None or not None at the same time"
            )

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return datasets.folder.has_file_allowed_extension(
                    x, cast(Tuple[str, ...], extensions)
                )

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        path_list = sorted(
            list((directory / "metadata").glob("*.npy")),
            key=lambda path: int(path.stem),
        )
        if not path_list:
            raise FileNotFoundError(f"No metadata found at {str(directory)}")
        # Get the 'pathfinder32/curv_baseline part of data_dir
        data_dir_stem = Path().joinpath(*directory.parts[-2:])
        instances = []
        for metadata_file in path_list:
            with open(metadata_file, "r") as f:
                for metadata in f.read().splitlines():
                    metadata = metadata.split()
                    image_path = Path(metadata[0]) / metadata[1]
                    if (
                        is_valid_file(str(image_path))
                        and str(data_dir_stem / image_path) not in PATHFINDER_BLACKLIST
                    ):
                        label = int(metadata[3])
                        instances.append((str(directory / image_path), label))
        return instances


class PathFinderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        test_batch_size,
        data_type,
        num_workers,
        pin_memory,
        resolution,
        level="hard",
        val_split=0.1,
        test_split=0.1,
        **kwargs,
    ):
        super().__init__()

        assert resolution in [32, 64, 128, 256]
        assert level in ["easy", "intermediate", "hard"]

        level_dir = {
            "easy": "curv_baseline",
            "intermediate": "curv_contour_length_9",
            "hard": "curv_contour_length_14",
        }[level]

        # Save parameters to self
        data_dir = (
            utils.get_original_cwd()
            + data_dir
            + f"/lra_release/pathfinder{resolution}/{level_dir}"
        )
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.resolution = resolution
        self.level = level

        self.val_split = val_split
        self.test_split = test_split

        # Determine data_type
        if data_type == "default":
            self.data_type = "image"
            self.data_dim = 2
        elif data_type == "sequence":
            self.data_type = data_type
            self.data_dim = 1
        else:
            raise ValueError(f"data_type {data_type} not supported.")

        # Determine sizes of dataset
        self.input_channels = 1
        self.output_channels = 2

        # Create transforms
        train_transform = [
            transforms.ToTensor(),
        ]
        # add augmentations
        if kwargs["augment"]:
            raise NotImplementedError

        self.train_transform = transforms.Compose(train_transform)

    def prepare_data(self):
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"""
            Directory {self.data_dir} not found.
            To get the dataset, download lra_release.gz from
            https://github.com/google-research/long-range-arena,
            then unzip it with tar -xvf lra_release.gz.
            Then point data_dir to the directory that contains pathfinderX, where X is the
            resolution (either 32, 64, 128, or 256).
            """
            )

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        # [2021-08-18] TD: I ran into RuntimeError: Too many open files.
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy("file_system")
        dataset = PathFinderDataset(self.data_dir, transform=self.train_transform)
        len_dataset = len(dataset)
        val_len = int(self.val_split * len_dataset)
        test_len = int(self.test_split * len_dataset)
        train_len = len_dataset - val_len - test_len
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(getattr(self, "seed", 42)),
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

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if self.data_type == "sequence":
            # If sequential, flatten the input [B, C, Y, X] -> [B, C, -1]
            x, y = batch
            x_shape = x.shape
            # Flatten
            x = x.view(x_shape[0], x_shape[1], -1)
            batch = x, y
        return batch
