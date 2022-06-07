# torch
import torch
import pytorch_lightning as pl
import glob
import hydra
import torchmetrics

# project
from optim import construct_optimizer, construct_scheduler
import ckconv

# logger
import wandb

# typing
from omegaconf import OmegaConf


class LightningWrapperBase(pl.LightningModule):
    def __init__(
        self,
        network: torch.nn.Module,
        cfg: OmegaConf,
    ):
        super().__init__()
        # Define network
        self.network = network
        # Save optimizer & scheduler parameters
        self.optim_cfg = cfg.optimizer
        self.scheduler_cfg = cfg.scheduler
        # Regularization metrics
        if self.optim_cfg.weight_decay != 0.0:
            self.weight_regularizer = ckconv.nn.LnLoss(
                weight_loss=self.optim_cfg.weight_decay,
                norm_type=2,
            )
        else:
            self.weight_regularizer = None
        # Placeholders for logging of best train & validation values
        self.no_params = -1
        # Explicitly define whether we are in distributed mode.
        self.distributed = cfg.train.distributed and cfg.train.avail_gpus != 1

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        # Construct optimizer & scheduler
        optimizer = construct_optimizer(
            model=self,
            optim_cfg=self.optim_cfg,
        )
        scheduler = construct_scheduler(
            optimizer=optimizer,
            scheduler_cfg=self.scheduler_cfg,
        )
        # Construct output dictionary
        output_dict = {"optimizer": optimizer}
        if scheduler is not None:
            output_dict["lr_scheduler"] = {}
            output_dict["lr_scheduler"]["scheduler"] = scheduler
            output_dict["lr_scheduler"]["interval"] = "step"

            # If we use a ReduceLROnPlateu scheduler, we must monitor val/acc
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.scheduler_cfg.mode == "min":
                    output_dict["lr_scheduler"]["monitor"] = "val/loss"
                else:
                    output_dict["lr_scheduler"]["monitor"] = "val/acc"
                output_dict["lr_scheduler"]["reduce_on_plateau"] = True
                output_dict["lr_scheduler"]["interval"] = "epoch"

            # TODO: ReduceLROnPlateau with warmup
            if isinstance(
                scheduler, ckconv.nn.scheduler.ChainedScheduler
            ) and isinstance(
                scheduler._schedulers[-1], torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                raise NotImplementedError("cannot use ReduceLROnPlateau with warmup")
        # Return
        return output_dict

    def on_train_start(self):
        if self.global_rank == 0:
            # Calculate and log the size of the model
            if self.no_params == -1:
                with torch.no_grad():
                    # Log parameters
                    no_params = ckconv.utils.no_params(self.network)
                    self.logger.experiment.summary["no_params"] = no_params
                    self.no_params = no_params

                    # Log code
                    code = wandb.Artifact(
                        f"source-code-{self.logger.experiment.name}", type="code"
                    )
                    # Get paths
                    paths = glob.glob(
                        hydra.utils.get_original_cwd() + "/**/*.py",
                        recursive=True,
                    )
                    paths += glob.glob(
                        hydra.utils.get_original_cwd() + "/**/*.yaml",
                        recursive=True,
                    )
                    # Filter paths
                    paths = list(filter(lambda x: "outputs" not in x, paths))
                    paths = list(filter(lambda x: "wandb" not in x, paths))
                    # Get all source files
                    for path in paths:
                        code.add_file(
                            path,
                            name=path.replace(f"{hydra.utils.get_original_cwd()}/", ""),
                        )
                    # Use the artifact
                    if not self.logger.experiment.offline:
                        wandb.run.use_artifact(code)


class ClassificationWrapper(LightningWrapperBase):
    def __init__(
        self,
        network: torch.nn.Module,
        cfg: OmegaConf,
        **kwargs,
    ):
        super().__init__(
            network=network,
            cfg=cfg,
        )
        # Other metrics
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        # Binary problem?
        self.multiclass = network.out_layer.out_channels != 1
        # Loss metric
        if self.multiclass:
            self.loss_metric = torch.nn.CrossEntropyLoss()
        else:
            self.loss_metric = torch.nn.BCEWithLogitsLoss()  # TODO: Required?
        # Function to get predictions:
        if self.multiclass:
            self.get_predictions = self.multiclass_prediction
        else:
            self.get_predictions = self.binary_prediction
        # Placeholders for logging of best train & validation values
        self.best_train_acc = 0.0
        self.best_val_acc = 0.0

    def _step(self, batch, accuracy_calculator):
        x, labels = batch
        logits = self(x)
        # Predictions
        predictions = self.get_predictions(logits)
        # Calculate accuracy and loss
        accuracy_calculator(predictions, labels)
        # For binary classification, the labels must be float
        if not self.multiclass:
            labels = labels.float()  # N
            logits = logits.view(-1)  # N
        loss = self.loss_metric(logits, labels)
        # Return predictions and loss
        return predictions, logits, loss

    def training_step(self, batch, batch_idx):
        # Perform step
        predictions, logits, loss = self._step(batch, self.train_acc)
        # Add regularization
        if self.weight_regularizer is not None:
            reg_loss = self.weight_regularizer(self.network)
        else:
            reg_loss = 0.0
        # Log and return loss (Required in training step)
        self.log(
            "train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=self.distributed
        )
        self.log(
            "train/acc",
            self.train_acc,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        self.log(
            "train/reg_loss",
            reg_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        return {"loss": loss + reg_loss, "logits": logits.detach()}

    def validation_step(self, batch, batch_idx):
        # Perform step
        predictions, logits, loss = self._step(batch, self.val_acc)
        # Log and return loss (Required in training step)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        self.log(
            "val/acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        return logits  # used to log histograms in validation_epoch_step

    def test_step(self, batch, batch_idx):
        # Perform step
        predictions, _, loss = self._step(batch, self.test_acc)
        # Log and return loss (Required in training step)
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        self.log(
            "test/acc",
            self.test_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )

    def training_epoch_end(self, train_step_outputs):
        flattened_logits = torch.flatten(
            torch.cat([step_output["logits"] for step_output in train_step_outputs])
        )
        self.logger.experiment.log(
            {
                "train/logits": wandb.Histogram(flattened_logits.to("cpu")),
                "global_step": self.global_step,
            }
        )
        # Log best accuracy
        train_acc = self.trainer.callback_metrics["train/acc_epoch"]
        if train_acc > self.best_train_acc:
            self.best_train_acc = train_acc.item()
            self.logger.experiment.log(
                {
                    "train/best_acc": self.best_train_acc,
                    "global_step": self.global_step,
                }
            )

    def validation_epoch_end(self, validation_step_outputs):
        # Gather logits from validation set and construct a histogram of them.
        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log(
            {
                "val/logits": wandb.Histogram(flattened_logits.to("cpu")),
                "val/logit_max_abs_value": flattened_logits.abs().max().item(),
                "global_step": self.global_step,
            }
        )
        # Log best accuracy
        val_acc = self.trainer.callback_metrics["val/acc"]
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc.item()
            self.logger.experiment.log(
                {
                    "val/best_acc": self.best_val_acc,
                    "global_step": self.global_step,
                }
            )

    @staticmethod
    def multiclass_prediction(logits):
        return torch.argmax(logits, 1)

    @staticmethod
    def binary_prediction(logits):
        return (logits > 0.0).squeeze().long()


class RegressionWrapper(LightningWrapperBase):
    def __init__(
        self,
        network: torch.nn.Module,
        cfg: OmegaConf,
        metric: str,
        **kwargs,
    ):
        super().__init__(
            network=network,
            cfg=cfg,
        )
        if metric == "MAE":
            MetricClass = torchmetrics.MeanAbsoluteError
            LossMetricClass = torch.nn.L1Loss
        elif metric == "MSE":
            MetricClass = torchmetrics.MeanSquaredError
            LossMetricClass = torch.nn.MSELoss
        else:
            raise ValueError(f"Metric {metric} not recognized")

        # Other metrics
        self.train_metric = MetricClass()
        self.val_metric = MetricClass()
        self.test_metric = MetricClass()
        # Loss metric
        self.loss_metric = LossMetricClass()
        # Placeholders for logging of best train & validation values
        self.best_train_loss = 1e9
        self.best_val_loss = 1e9

    def _step(self, batch, metric_calculator):
        x, labels = batch
        prediction = self(x)
        # Calculate loss
        metric_calculator(prediction, labels)
        loss = self.loss_metric(prediction, labels)
        # Return predictions and loss
        return prediction, loss

    def training_step(self, batch, batch_idx):
        # Perform step
        _, loss = self._step(batch, self.train_metric)
        # Log loss
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)

        # Add regularization
        if self.regularizer is None:
            # Return loss (required in training step)
            return loss
        else:
            reg_loss = self.regularizer(self.network)
            self.log("train/reg_loss", reg_loss, on_epoch=True, prog_bar=True)
            return loss + reg_loss

    def validation_step(self, batch, batch_idx):
        # Perform step
        predictions, loss = self._step(batch, self.val_metric)
        # Log and return loss (Required in training step)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Perform step
        predictions, loss = self._step(batch, self.test_metric)
        # Log and return loss (Required in training step)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def training_epoch_end(self, train_step_outputs):
        # Log best accuracy
        train_loss = self.trainer.callback_metrics["train/loss_epoch"]
        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss.item()
            self.logger.experiment.log(
                {
                    "train/best_loss": self.best_train_loss,
                    "global_step": self.global_step,
                }
            )

    def validation_epoch_end(self, validation_step_outputs):
        # Log best accuracy
        val_loss = self.trainer.callback_metrics["val/loss"]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            self.logger.experiment.log(
                {
                    "val/best_loss": self.best_val_loss,
                    "global_step": self.global_step,
                }
            )


#############################
#    Point Cloud Models     #
#############################
class PyGClassificationWrapper(ClassificationWrapper):
    def _step(self, batch, accuracy_calculator):
        logits = self(batch)
        # Predictions
        predictions = torch.argmax(logits, 1)
        # Calculate accuracy and loss
        accuracy_calculator(predictions, batch.y)
        loss = self.loss_metric(logits, batch.y)
        # Return predictions and loss
        return predictions, logits, loss


class PyGRegressionWrapper(RegressionWrapper):
    def _step(self, batch, metric_calculator):
        prediction = self(batch)
        # Calculate loss
        metric_calculator(prediction, batch.y)
        loss = self.loss_metric(prediction, batch.y)
        # Return predictions and loss
        return prediction, loss
