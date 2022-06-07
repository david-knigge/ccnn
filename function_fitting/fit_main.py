# Used to run function fitting experiments

import copy
import os
import numpy as np

# torch
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl

# project
import ckconv
from target_functions import construct_target_function
from ckconv.utils.grids import linspace_grid

# config
import hydra
from omegaconf import OmegaConf

# logging
import wandb


@hydra.main(config_path=".", config_name="config.yaml")
def main(
    cfg: OmegaConf,
):
    # We possibly want to add fields to the config file. Thus, we set struct to False.
    OmegaConf.set_struct(cfg, False)
    validate_cfg(cfg)

    # Set seed
    # IMPORTANT! This does not make training entirely deterministic! Trainer(deterministic=True) required!
    pl.seed_everything(cfg.seed, workers=True)

    # Construct target function
    targets = construct_target_function(cfg)
    # Construct input grid
    x = linspace_grid(grid_sizes=torch.tensor(cfg.target.length).repeat(cfg.target.dim))

    # Construct kernel
    kernel = construct_kernel(cfg)

    # Add batch dimension and send to device
    kernel.to(device=cfg.device)
    targets = targets.to(device=cfg.device)
    x = x.unsqueeze(0).to(device=cfg.device)

    # Construct optimizer
    optimizer = construct_optimizer(kernel, cfg)

    # Start wandb
    if cfg.debug:
        os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=ckconv.utils.flatten_configdict(cfg),
        group=cfg.target.type,
        save_code=True,
    )

    # Fit the kernel
    if cfg.train.do:

        kernel.train()
        log_interval = cfg.train.log_interval
        current_target = 0
        global_step = 0

        while current_target < cfg.target.count:

            # Select target
            target = targets[current_target].unsqueeze(0)
            # Log the target
            log_target(target, cfg, step=global_step)
            # Train vars
            iter = 0
            best_loss_train = 1e9

            while best_loss_train > cfg.train.loss_threshold:
                optimizer.zero_grad()

                fit = kernel(x)
                loss = torch.nn.functional.mse_loss(fit, target)

                loss.backward()
                optimizer.step()

                if loss.item() < best_loss_train:
                    best_loss_train = loss.item()
                    wandb.run.summary[
                        f"best_loss_target_{current_target}"
                    ] = best_loss_train
                    # best_weights = copy.deepcopy(kernel.state_dict())

                if iter % cfg.plot.interval == 0:
                    log_fit(fit, loss, cfg, step=global_step)

                if iter % log_interval == 0:
                    # Report and log
                    print(
                        f"Iter: {iter:2d}/{cfg.train.no_iters:6d} \t lr: {cfg.optimizer.lr:.4f}\tLoss: {loss.item():.6f}"
                    )
                    # print(f"PSNR: {psnr}")

                    wandb.log({"mse_loss": loss.item()}, step=global_step)
                    # wandb.log({"psnr": psnr.item()}, step=iter)

                if iter == cfg.train.no_iters:
                    break
                iter += 1
                global_step += 1

            log_fit(fit, loss, cfg, logger_key="best_fit", step=global_step)

            # Save the required iterations
            wandb.run.summary[f"time_target_{current_target}"] = iter

            current_target += 1
            # # save best weights and load them
            # save_weights_to_wandb(best_weights, "best_weights")
            # kernel.load_state_dict(best_weights)

    wandb.run.summary["total_time"] = global_step

    # Check the fit
    # kernel.eval()
    # fit = kernel(x)
    # loss = torch.nn.functional.mse_loss(fit, target)
    # wandb.run.summary["best_mse"] = loss.item()
    # log_fit(fit, loss, cfg, logger_key="best_fit")

    # Finish
    return


def construct_kernel(cfg):
    # Gather kernel nonlinear and norm type
    kernel_norm = getattr(torch.nn, cfg.kernel.norm)
    kernel_nonlinear = getattr(torch.nn, cfg.kernel.nonlinearity)

    # Construct kernel
    KernelClass = getattr(ckconv.nn.ck, cfg.kernel.type)
    kernel = KernelClass(
        data_dim=cfg.target.dim,
        out_channels=cfg.target.no_channels,
        hidden_channels=cfg.kernel.no_hidden,
        no_layers=cfg.kernel.no_layers,
        bias=cfg.kernel.bias,
        causal=False,
        # MFN
        omega_0=cfg.kernel.omega_0,
        steerable=False,  # TODO
        init_spatial_value=1.0,
        # SIREN
        learn_omega_0=False,  # TODO
        # MLP & RFNet
        NonlinearType=kernel_nonlinear,
        NormType=kernel_norm,
        weight_norm=False,
    )
    return kernel


def construct_optimizer(
    model,
    cfg: OmegaConf,
):
    params = model.parameters()
    # Construct optimizer
    if cfg.optimizer.name == "SGD":
        # Unpack values from optim_cfg.params
        momentum = cfg.optimizer.momentum
        nesterov = cfg.optimizer.nesterov

        optimizer = torch.optim.SGD(
            params=params,
            lr=cfg.optimizer.lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=0.0,
        )
    else:
        optimizer = getattr(torch.optim, cfg.optimizer.name)(
            params=params,
            lr=cfg.optimizer.lr,
            weight_decay=0.0,
        )
    return optimizer


def save_weights_to_wandb(weights, name):
    filename = f"{name}.pt"
    path = os.path.join(wandb.run.dir, filename)
    # Save locally
    torch.save({"state_dict": weights}, path)
    # Call wandb to save the object, syncing it directly
    wandb.save(path)


def log_target(
    target,
    cfg,
    step=0,
):
    target = target[0].detach().cpu().numpy()[: cfg.plot.no_channels]

    fig = plt.figure()
    if cfg.target.dim == 1:
        ax = fig.add_subplot(111)
        for i in range(target.shape[0]):
            ax.plot(target[i])
    elif cfg.target.dim == 2:
        ax = fig.add_subplot(111)
        if cfg.plot.no_channels == 3:
            target = target.transpose(1, 2, 0)
        elif cfg.plot.no_channels == 1:
            target = target[0]
        ax.imshow(target)
    else:
        ax = fig.add_subplot(111, projection="3d")
        x, y, z = np.indices(
            (target.shape[1], target.shape[2], target.shape[3])
        ) / float(cfg.target.length - 1)
        ax.voxels(x, y, z, facecolors=target.transpose(1, 2, 3, 0).repeat(3, 3))
        ax.set_zticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    # show or log
    if cfg.debug:
        fig.show()
    else:
        wandb.log({"target": wandb.Image(fig)}, step=step)
    # close and return
    plt.close("all")
    return


def log_fit(
    fit,
    loss,
    cfg,
    logger_key="fit",
    step=0,
):
    fit = fit[0].detach().cpu().numpy()[: cfg.plot.no_channels]

    fig = plt.figure()
    if cfg.target.dim == 1:
        ax = fig.add_subplot(111)
        for i in range(fit.shape[0]):
            ax.plot(fit[i])
    elif cfg.target.dim == 2:
        ax = fig.add_subplot(111)
        if cfg.plot.no_channels == 3:
            fit = fit.transpose(1, 2, 0)
        elif cfg.plot.no_channels == 1:
            fit = fit[0]
        ax.imshow(fit)
    elif cfg.target.dim == 3:
        ax = fig.axes(projection="3d")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.text(
        0.99,
        0.015,
        f"MSE: {loss.item():.3e}",
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=ax.transAxes,
        color="Black",
        fontsize=9,
        weight="roman",
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 4},
    )
    fig.tight_layout()
    # show or log
    if cfg.debug:
        fig.show()
    else:
        wandb.log({logger_key: wandb.Image(fig)}, step=step)
    # close & return
    plt.close("all")
    return


def validate_cfg(cfg):
    if cfg.target.dim == 3 and cfg.plot.no_channels != 1:
        raise ValueError(
            f"In 3D only one channel can be plotted. Current: {cfg.plot.no_channels}"
        )
    if cfg.target.dim == 2 and cfg.plot.no_channels not in [1, 3]:
        raise ValueError(
            f"In 2D only 1 and 3 channels can be plotted. Current: {cfg.plot.no_channels}"
        )


if __name__ == "__main__":
    main()
