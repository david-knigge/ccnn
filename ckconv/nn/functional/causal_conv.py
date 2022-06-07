import torch
import torch.nn.functional as f
import torch.fft

from typing import Optional


def causal_padding(
    x: torch.Tensor,
    kernel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 1. Pad the input signal & kernel tensors.
    # Check if the size of the kernel is odd. If not, add a pad of zero to make them odd.
    if kernel.shape[-1] % 2 == 0:
        kernel = f.pad(kernel, [1, 0], value=0.0)
        # x = torch.nn.functional.pad(x, [1, 0], value=0.0)
    # 2. Perform padding on the input so that output equals input in length
    x = f.pad(x, [kernel.shape[-1] - 1, 0], value=0.0)
    return x, kernel


def padding(
    x: torch.Tensor,
    kernel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 1. Pad the input signal & kernel tensors.
    # x = torch.nn.functional.pad(x, [1, 0], value=0.0)
    # 2. Perform padding on the input so that output equals input in length
    x = f.pad(x, [kernel.shape[-1] // 2, kernel.shape[-1] // 2], value=0.0)
    return x, kernel


def conv1d(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    separable: bool = False,
    causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Args:
        x: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Optional, Tensor) Bias tensor to add to the output.
        padding: (int) Number of zero samples to pad the input on the last dimension.
    Returns:
        (Tensor) Convolved tensor
    """
    # Pad
    if causal:
        x, kernel = causal_padding(x, kernel)
    else:
        x, kernel = padding(x, kernel)
    # Define groups for (non)separable.
    if separable:
        groups = kernel.shape[1]
        kernel = kernel.view(kernel.shape[1], 1, -1)
    else:
        groups = 1
    # Perform convolution
    return torch.nn.functional.conv1d(x, kernel, bias=bias, padding=0, groups=groups)


def fftconv1d(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    separable: bool = False,
    causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Args:
        x: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Optional, Tensor) Bias tensor to add to the output.
        padding: (int) Number of zero samples to pad the input on the last dimension.
    Returns:
        (Tensor) Convolved tensor
    """

    x_shape = x.shape

    # 1. Handle padding
    if causal:
        x, kernel = causal_padding(x, kernel)
    else:
        x, kernel = padding(x, kernel)

    # 2. Pad the kernel tensor to make them equally big. Required for fft.
    kernel = f.pad(kernel, [0, x.size(-1) - kernel.size(-1)])

    # 3. Perform fourier transform
    x_fr = torch.fft.rfft(x, dim=-1)
    kernel_fr = torch.conj(torch.fft.rfft(kernel, dim=-1))

    # 4. Multiply the transformed matrices:
    # (Input * Conj(Kernel)) = Correlation(Input, Kernel)
    if separable:
        output_fr = kernel_fr * x_fr
    else:
        output_fr = torch.einsum("bi..., oi... -> bo...", x_fr, kernel_fr)

    # 5. Compute inverse FFT, and remove extra padded values
    # Once we are back in the spatial domain, we can go back to float precision, if double used.
    out = torch.fft.irfft(output_fr, dim=-1)[..., : x_shape[-1]]

    # 6. Optionally, add a bias term before returning.
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    return out
