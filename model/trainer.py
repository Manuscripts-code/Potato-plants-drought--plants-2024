import shutil
import tempfile
from abc import abstractmethod
from pathlib import Path

import keyboard
import mlflow
import numpy as np
import torch
from numpy import inf
from torchvision.utils import make_grid

from configs import configs
from utils.utils import ensure_dir, inf_loop, write_json, write_txt

from .helpers import MetricTracker, TensorboardWriter


class OutputHook(list):
    def __call__(self, module, input, output):
        self.append(output)


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1
        self.stop_flag = False
        self.checkpoints_dir, self.save_dir = None, None

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.save_dir, self.logger, cfg_trainer["tensorboard"])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        self.save_dir = self._set_save_dir()
        self.checkpoints_dir = ensure_dir(self.save_dir / "artifacts/checkpoints")
        self._save_config()

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # log metrics with mlflow
            mlflow.log_metrics(result, step=log["epoch"])

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best) or (
                        self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best
                    )
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(self.mnt_metric)
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

                if self.stop_flag:
                    self.logger.info("Keyboard interrupt. Training stops.")
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _set_save_dir(self):
        run_id = mlflow.active_run().info.run_id
        experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
        save_dir = Path(configs.MODEL_REGISTRY, experiment_id, run_id)
        return save_dir

    def _save_config(self):
        """
        Saving artifacts
        """
        with tempfile.TemporaryDirectory() as dp:
            ensure_dir(Path(dp) / "configs")
            write_json(self.config.config, Path(dp, "configs/config.json"))
            write_txt(str(self.model), Path(dp, "configs/model_arch.txt"))
            shutil.copyfile(configs.__file__, Path(dp, "configs/configs.py"))
            mlflow.log_artifacts(dp)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoints_dir / "checkpoint-epoch{}.pth".format(epoch))
        # torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoints_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path / "artifacts/checkpoints" / "model_best.pth")
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that of "
                "checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint["config"]["optimizer"]["type"] != self.config["optimizer"]["type"]:
            self.logger.warning(
                "Warning: Optimizer type given in config file is different from that of checkpoint. "
                "Optimizer parameters not being resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer
        )

        # change validation monitoring metric to train metric
        if not self.do_validation and "val_" in self.mnt_metric:
            self.mnt_metric = self.mnt_metric.replace("val_", "")

        self.output_hook = OutputHook()
        self.model.spectral.act2.register_forward_hook(self.output_hook)
        self.l1_lambda = 0

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target, _) in enumerate(self.data_loader):
            # convert to column vector
            target = target.view(len(target), -1).float()
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            pred_class = self.model(data)
            loss = self.criterion("bce")(pred_class, target)  # classification loss
            l1_penalty = self.criterion("l1")(self.output_hook[0])  # l1 penalty on 1d band scaler vector

            loss = loss + l1_penalty * self.l1_lambda
            loss.backward()
            self.optimizer.step()
            self.output_hook.clear()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(pred_class, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss.item()
                    )
                )

            if batch_idx == self.len_epoch:
                break
            # Uncomment if you would like to end training before the end
            # if keyboard.is_pressed("e"):
            #     self.logger.info("You pressed <e> to exit training.")
            #     self.stop_flag = True

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target, _) in enumerate(self.valid_data_loader):
                # convert to column vector
                target = target.view(len(target), -1).float()
                data, target = data.to(self.device), target.to(self.device)

                pred_class = self.model(data)
                loss = self.criterion("bce")(pred_class, target)  # classification loss
                l1_penalty = self.criterion("l1")(
                    self.output_hook[0]
                )  # l1 penalty on 1d band scaler vector
                loss = loss + l1_penalty * self.l1_lambda

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid")
                self.valid_metrics.update("loss", loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(pred_class, target))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
