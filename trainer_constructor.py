import pytorch_lightning as pl

# typing
import torch.cuda
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger


def construct_trainer(
    cfg: OmegaConf,
    wandb_logger: pl.loggers.WandbLogger,
) -> tuple[pl.Trainer, pl.Callback]:

    # Set up precision
    if cfg.train.mixed_precision:
        precision = 16
    else:
        precision = 32

    # Set up determinism
    if cfg.deterministic:
        deterministic = True
        benchmark = False
    else:
        deterministic = False
        benchmark = True

    # Callback to print model summary
    modelsummary_callback = pl.callbacks.ModelSummary(
        max_depth=-1,
    )

    # Metric to monitor
    if cfg.scheduler.mode == "max":
        monitor = "val/acc"
    elif cfg.scheduler.mode == "min":
        monitor = "val/loss"

    # Callback for model checkpointing:
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        mode=cfg.scheduler.mode,  # Save on best validation accuracy
        save_last=True,  # Keep track of the model at the last epoch
        verbose=True,
    )

    # Callback for learning rate monitoring
    lrmonitor_callback = pl.callbacks.LearningRateMonitor()

    # Callback for early stopping:
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor=monitor,
        mode=cfg.scheduler.mode,
        patience=cfg.train.max_epochs_no_improvement,
        verbose=True,
    )

    """
    TODO:
    detect_anomaly
    limit batches
    profiler
    overfit_batches
    resume from checkpoint
    StochasticWeightAveraging
    log_every_n_steps
    """
    # Distributed training params
    if cfg.device == "cuda":
        sync_batchnorm = cfg.train.distributed
        strategy = "ddp_find_unused_parameters_false" if cfg.train.distributed else None
        gpus = cfg.train.avail_gpus if cfg.train.distributed else 1
        num_nodes = cfg.train.num_nodes if (cfg.train.num_nodes != -1) else 1
    else:
        gpus = 0
        sync_batchnorm = False
        strategy = None
        num_nodes = 1

    # create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        logger=wandb_logger,
        gradient_clip_val=cfg.train.grad_clip,
        accumulate_grad_batches=cfg.train.accumulate_grad_steps,
        track_grad_norm=cfg.train.track_grad_norm,
        # Callbacks
        callbacks=[
            modelsummary_callback,
            lrmonitor_callback,
            checkpoint_callback,
            early_stopping_callback,
        ],
        # Multi-GPU
        num_nodes=num_nodes,
        gpus=gpus,
        strategy=strategy,
        sync_batchnorm=sync_batchnorm,
        # auto_select_gpus=True,
        # Precision
        precision=precision,
        # Determinism
        deterministic=deterministic,
        benchmark=benchmark,
    )
    return trainer, checkpoint_callback
