from itertools import chain
from pathlib import Path

import numpy as np
import PIL
import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from tts.base import BaseTrainer
from tts.datasets.MelSpectrogram import MelSpectrogram
from tts.logger.utils import plot_spectrogram_to_buf
from tts.metrics import EER
from tts.utils import ROOT_PATH, MetricTracker, inf_loop
from tts.utils.util import MelSpectrogramConfig


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics_train,
            metrics_test,
            optimizer,
            config,
            device,
            log_step,
            dataloader,
            scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(
            model,
            criterion,
            metrics_train,
            metrics_test,
            optimizer,
            config,
            device,
        )
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloader['train']
        self.eval_dataloaders = {key: value for key, value in dataloader.items()}
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.scheduler = scheduler
        self.log_step = log_step
        self.wav_to_mel = MelSpectrogram(MelSpectrogramConfig()).to(device)
        self.metrics = [
                           "loss",
                       ] + [m.name for m in self._metrics_train]
        self.train_metrics = MetricTracker(
            "grad norm", "EER",
            *self.metrics,
            writer=self.writer,
        )
        self.eval_metrics = MetricTracker(
            *self.metrics,
            writer=self.writer,
        )
        self.eer_metric = EER()

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        batch["audio"] = batch["audio"].to(device)
        batch["is_real"] = batch["is_real"].to(device)
        return batch

    def _clip_grad_norm(self, params=None):
        if params is None:
            params = self.model.parameters()
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(params, self.config["trainer"]["grad_norm_clip"])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    f"Train Epoch: {epoch} Loss: {batch['loss']} "
                )
                self.writer.add_scalar(
                    "learning rate", self.optimizer.param_groups[0]["lr"],
                )
                self._log_scalars(self.train_metrics)
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
                if batch_idx >= self.len_epoch:
                    break
        log = last_train_metrics

        for part, dataloader in self.eval_dataloaders.items():
            eval_res = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in eval_res.items()})
        return log

    def process_batch(
            self, batch, is_train: bool, metrics: MetricTracker
    ):
        batch = self.move_batch_to_device(batch, self.device)

        batch["pred_spoof"] = self.model(batch["audio"])
        batch["loss"] = self.criterion(batch["pred_spoof"], batch["is_real"])

        if is_train:
            self.optimizer.zero_grad()
            batch["loss"].backward()
            self._clip_grad_norm(self.model.parameters())
            self.optimizer.step()
            metrics.update(
                "grad norm",
                self.get_grad_norm(self.model.parameters()),
            )
        metrics.update("loss", batch["loss"].item())
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.eval_metrics.reset()
        logits = []
        targets = []
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.eval_metrics,
                )
                logits += list(batch['pred_spoof'][:, 1].detach().cpu().numpy())
                targets += list(batch['is_real'].detach().cpu().numpy().astype(bool))
            logits = np.array(logits)
            targets = np.array(targets)
            eer, thresh = self.eer_metric(targets, logits)
            self.writer.add_scalar("EER", eer)
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.eval_metrics)
            if "val" in part:
                torch.save(self.model.state_dict(), ROOT_PATH / f"outputs/{epoch}.pth")
                print("Save after val")
            # self._log_predictions(is_validation=True, **batch)
            # self._log_spectrogram(batch["spectrogram"])
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        print(f"part: {part}: EER: {eer} with thresh: {thresh}")
        ans = self.eval_metrics.result() | {"EER": eer, "thesh": eer}
        return ans

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log(self, true_mels, pred_mels, true_audio, pred_audio):
        idx = np.random.choice(np.arange(len(true_mels)))
        img_true = PIL.Image.open(
            plot_spectrogram_to_buf(true_mels[idx].detach().cpu().numpy().T)
        )
        img_pred = PIL.Image.open(
            plot_spectrogram_to_buf(pred_mels[idx].detach().cpu().numpy().T)
        )
        self.writer.add_image("Target mel", ToTensor()(img_true))
        self.writer.add_image("Prediction mel", ToTensor()(img_pred))
        self.writer.add_audio("Target audio", true_audio.squeeze(), sample_rate=22050)
        self.writer.add_audio(
            "Prediction audio", pred_audio.squeeze(), sample_rate=22050
        )

    @torch.no_grad()
    def get_grad_norm(self, parameters, norm_type=2):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
