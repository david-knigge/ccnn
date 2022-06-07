import torch
import torchvision as vision
import torchaudio as audio
from hydra import utils
from pathlib import Path


def construct_target_function(cfg):
    return globals()[cfg.target.type](cfg)


def random(cfg):
    target = torch.rand(
        [
            cfg.target.count,
            cfg.target.no_channels,
            *(cfg.target.length,) * cfg.target.dim,
        ],
    )
    return target


# Returns either audio (dim=1), images (dim=2) or 3d images (dim=3)
def natural(cfg):
    root = Path(utils.get_original_cwd()).parent / "data"
    if cfg.target.dim == 1:
        # Download dataset
        dataset = audio.datasets.SPEECHCOMMANDS(
            root=str(root),
            subset="training",
            download=True,
        )

        def audio_collate(batch):
            xs = torch.stack(
                [
                    torch.nn.functional.pad(
                        datapoint[0], (16000 - datapoint[0].shape[-1], 0)
                    )
                    for datapoint in batch
                ],
                dim=0,
            )
            return xs

        # Create loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.target.count,
            shuffle=True,
            num_workers=1,
            collate_fn=audio_collate,
        )
        # Gather first config.no_samples images from the dataset
        for data in loader:
            target = data.clone()[..., -cfg.target.length :]
            target = 1 / target.abs().max().item() * target
            break
    if cfg.target.dim == 2:
        # Define transforms
        transform = []
        if cfg.target.no_channels == 1:
            transform += [
                vision.transforms.Grayscale(),
            ]
        transform += [
            vision.transforms.Resize(cfg.target.length),
            vision.transforms.ToTensor(),
        ]
        transform = vision.transforms.Compose(transform)
        # Download dataset
        dataset = vision.datasets.STL10(
            root=str(root),
            split="train",
            transform=transform,
            download=True,
        )
        # Create loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.target.count,
            shuffle=True,
            num_workers=1,
        )
        # Gather first config.no_samples images from the dataset
        for images, _ in loader:
            target = images.clone()
            break
    return target


def perlin(cfg):
    pass
