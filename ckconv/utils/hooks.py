import wandb
import torch


from ckconv.utils.visualisation import visualize_tensor_1d, visualize_tensor_2d


# def gradient_norm_hook(module, input, output, name):
#     """ Count where gradient is small. """
#     out = torch.flatten(input.detach(), start_dim=1)
#
#     # size of gradient
#     small_grad = 1e-5
#
#     # calculate norm per neuron
#     zero_indices = set(torch.all(torch.norm(out) < small_grad, dim=0).nonzero().flatten().numpy().tolist())
#
#     if not hasattr(module, 'small_grads'):
#         module.dead_indices = zero_indices
#     else:
#         module.dead_indices = zero_indices.intersection(module.dead_indices)


def count_dead_neurons_hook(module, module_in, module_out, name):
    """Count number of dead neurons by checking which outputs remain zero throughout the entire training set."""
    out = torch.flatten(module_out.detach(), start_dim=1)
    # indices to set for easy calculation of intersection

    zero_indices = set(torch.all(out == 0, dim=0).nonzero().flatten().numpy().tolist())

    if not hasattr(module, "dead_indices"):
        module.dead_indices = zero_indices
    else:
        module.dead_indices = zero_indices.intersection(module.dead_indices)


def log_dead_neuron_count_hook(module, module_in, module_out, name):
    """Log and remove dead neuron count"""
    out = torch.flatten(module_out.detach(), start_dim=1)
    # TODO log as dead/ nonlinearity in sequential/modulelist for naming consistency

    if hasattr(module, "dead_indices"):
        dead_count = len(module.dead_indices)

        if wandb.run:
            wandb.log({"dead/" + name: dead_count / out.shape[1]}, commit=False)

        del module.dead_indices


def module_out_hist_hook(module, module_in, module_out, output, name):
    """Logs histogram of output values of a module."""

    # backward hooks get output in a tuple, see
    # https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks
    if type(module_out) == tuple:
        output = output[0]
    if type(module_in) == tuple:
        input = module_in[0]

    if wandb.run:
        wandb.log(
            {
                "activations/" + name: wandb.Histogram(output.detach().cpu().numpy()),
                # name + ".sum": output.sum().detach().cpu().numpy(),
            },
            commit=False,
        )


def visualize_kernel_out_hook(module, module_in, module_out, name):
    """Logs a visualisation of kernelnet output values (conv kernels) in a plotly figure format."""

    # Get conv kernel.
    kernel = module.conv_kernel

    # Get dimensionality of conv kernel.
    data_dim = module.data_dim

    if wandb.run:
        if data_dim == 1:
            figs = visualize_tensor_1d(kernel.detach().cpu())
        elif data_dim == 2:
            figs = visualize_tensor_2d(kernel.detach().cpu())
        wandb.log(
            {f"kernels/{name}.out_channel.{idx}": fig for idx, fig in enumerate(figs)},
            commit=False,
        )


def visualize_conv_kernel_out_hook(module, input, output, name):
    """Logs a visualisation of kernelnet output values (conv kernels) in a plotly figure format."""
    kernel = module.weight

    if wandb.run:
        figs = visualize_tensor_1d(kernel.detach().cpu())
        wandb.log(
            {f"kernels/{name}.out_channel.{idx}": fig for idx, fig in enumerate(figs)},
            commit=False,
        )


@torch.no_grad()
def log_mask_params(module, input, output, name):
    if wandb.run:
        # Log
        wandb.log(
            {f"mask/{name}_mean": module.mask_mean_param.item()},
            commit=False,
        )
        wandb.log(
            {f"mask/{name}_variance": module.mask_width_param.item()},
            commit=False,
        )


def get_statistics(tensor: torch.Tensor):
    return (
        tensor.max().item(),
        tensor.mean().item(),
        tensor.std().item(),
        tensor.min().item(),
        tensor.abs().max().item(),
    )


@torch.no_grad()
def log_output_statistics(module, input, output, name):
    if wandb.run:
        # Calculate statistics
        max, mean, std, min, max_abs = get_statistics(output)
        # Log
        for statistic in ["max", "mean", "std", "min", "max_abs"]:
            wandb.log(
                {f"outputs/{name}_{statistic}": eval(statistic)},
                commit=False,
            )
        # Histogram
        wandb.log(
            {
                f"outputs/{name}_histogram": wandb.Histogram(
                    output.flatten().detach().to("cpu")
                )
            },
            commit=False,
        )


@torch.no_grad()
def log_parameter_statistics(module, input, output, name):

    if wandb.run:
        for parameter in ["weight", "bias"]:
            # Calculate statistics
            max, mean, std, min, max_abs = get_statistics(getattr(module, parameter))
            # Log
            for statistic in ["max", "mean", "std", "min", "max_abs"]:
                wandb.log(
                    {f"params/{name}_{parameter}_{statistic}": eval(statistic)},
                    commit=False,
                )
            # Histogram
            wandb.log(
                {
                    f"params/{name}_{parameter}_histogram": wandb.Histogram(
                        module.weight.flatten().detach().to("cpu")
                    )
                },
                commit=False,
            )


@torch.no_grad()
def log_ckernel_statistics(module, input, output, name):
    """Logs statistics of kernelnet output values (conv kernels)."""
    if wandb.run:
        kernel = module.conv_kernel
        # Calculate statistics
        max, mean, std, min, max_abs = get_statistics(kernel)
        # Log
        for statistic in ["max", "mean", "std", "min", "max_abs"]:
            wandb.log(
                {f"kernels/{name}_{statistic}": eval(statistic)},
                commit=False,
            )
        # Histogram
        wandb.log(
            {
                f"kernels/{name}_histogram": wandb.Histogram(
                    kernel.flatten().detach().to("cpu")
                )
            },
            commit=False,
        )


def visualize_ckconv_out_hook(module, input, output, name):
    pass
