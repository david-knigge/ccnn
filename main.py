# torch
import torch
import pytorch_lightning as pl

# Project
import wandb

import ckconv
from dataset_constructor import construct_datamodule
from model_constructor import construct_model
from trainer_constructor import construct_trainer

from functools import partial
from hook_registration import register_hooks

# Loggers
from pytorch_lightning.loggers import WandbLogger

# Configs
import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="cfg", config_name="config.yaml")
def main(
    cfg: OmegaConf,
):
    # We possibly want to add fields to the config file. Thus, we set struct to False.
    OmegaConf.set_struct(cfg, False)

    # Set seed
    # IMPORTANT! This does not make training entirely deterministic! Trainer(deterministic=True) required!
    pl.seed_everything(cfg.seed, workers=True)

    # Check number of available gpus
    cfg.train.avail_gpus = torch.cuda.device_count()

    # Construct data_module
    datamodule = construct_datamodule(cfg)
    datamodule.prepare_data()
    datamodule.setup()

    # Append no of iteration to the cfg file for the definition of the schedulers
    distrib_batch_size = cfg.train.batch_size
    if cfg.train.distributed:
        distrib_batch_size *= cfg.train.avail_gpus
    cfg.scheduler.iters_per_train_epoch = (
        len(datamodule.train_dataset) // distrib_batch_size
    )
    cfg.scheduler.total_train_iters = (
        cfg.scheduler.iters_per_train_epoch * cfg.train.epochs
    )

    # Construct model
    model = construct_model(cfg, datamodule)

    # Initialize wandb logger
    if cfg.debug:
        log_model = False
        offline = True
    else:
        log_model = "all"
        offline = False
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=ckconv.utils.flatten_configdict(cfg),
        log_model=log_model,  # used to save models to wandb during training
        offline=offline,
        save_code=True,
    )

    # Before start training. Verify arguments in the cfg.
    verify_config(cfg)

    # Recreate the command that instantiated this run.
    if isinstance(wandb_logger.experiment.settings, wandb.Settings):
        args = wandb_logger.experiment.settings._args
        command = " ".join(args)

        # Log the command.
        wandb_logger.experiment.config.update({"command": command})

    # Print the cfg files prior to training
    print(f"Input arguments \n {OmegaConf.to_yaml(cfg)}")

    # Create trainer
    trainer, checkpoint_callback = construct_trainer(cfg, wandb_logger)

    # Load checkpoint
    if cfg.pretrained.load:
        # Construct artifact path.
        checkpoint_path = hydra.utils.get_original_cwd() + f"/artifacts/{cfg.pretrained.filename}"

        # Load model from artifact
        print(
            f'IGNORE this validation run. Required due to problem with Lightning model loading \n {"#" * 200}'
        )
        trainer.validate(model, datamodule=datamodule)
        print("#" * 200)
        checkpoint_path += "/model.ckpt"
        model = model.__class__.load_from_checkpoint(
            checkpoint_path,
            network=model.network,
            cfg=cfg,
        )

    # Test before training
    if cfg.test.before_train:
        trainer.validate(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)

    # register hooks
    if cfg.hooks_enabled:
        model.configure_callbacks = partial(register_hooks, cfg, model)

    # Train
    if cfg.train.do:
        if cfg.pretrained.load:
            # From preloaded point
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
        else:
            # From scratch
            trainer.fit(model=model, datamodule=datamodule)
        # Load state dict from best performing model
        model.load_state_dict(
            torch.load(checkpoint_callback.best_model_path)["state_dict"],
        )

    # Validate and test before finishing
    trainer.validate(
        model,
        datamodule=datamodule,
    )
    trainer.test(
        model,
        datamodule=datamodule,
    )


def verify_config(cfg: OmegaConf):
    if cfg.train.distributed and cfg.train.avail_gpus < 2:
        raise ValueError(
            f"Distributed only available with more than 1 GPU. Avail={cfg.train.avail_gpus}"
        )
    if cfg.conv.causal and cfg.net.data_dim != 1:
        raise ValueError("Causal conv is only supported in 1D.")
    if (
        cfg.conv.type in ["SeparableFlexConv", "FlexConv"]
        and cfg.mask.type != "gaussian"
        and cfg.net.data_dim != 1
    ):
        raise ValueError(f"Only gaussian masks are supported in {cfg.net.data_dim}.")
    if cfg.train.batch_size % cfg.train.accumulate_grad_steps:
        raise ValueError(
            f"Batch size must be divisible by the number of grad accumulation steps.\n"
            f"Values: batch_size:{cfg.train.batch_size}, "
            f"accumulate_grad_steps:{cfg.train.accumulate_grad_steps}",
        )


if __name__ == "__main__":
    main()
