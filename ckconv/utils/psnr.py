import numpy as np


def psnr(img1, img2):
    """Calculates PSNR between two images.
    Args:
        img1 (torch.Tensor):
        img2 (torch.Tensor):
    """
    return (
        20.0 * np.log10(1.0)
        - 10.0 * (img1 - img2).detach().pow(2).mean().log10().to("cpu").item()
    )
