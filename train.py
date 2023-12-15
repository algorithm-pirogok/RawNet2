import argparse
import collections
import itertools
import warnings

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from tts.trainer import Trainer
from tts.utils import get_logger, prepare_device
from tts.utils.object_loading import get_dataloaders

# Отключение предупреждений
warnings.simplefilter("ignore", UserWarning)

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


@hydra.main(config_path="tts/configs", config_name="main_config")
def main(clf: DictConfig):
    dataloader = get_dataloaders(clf["data"])

    model = instantiate(clf["arch"])

    logger = get_logger("train")
    # setup data_loader instances
    # build model architecture, then print to console
    logger.info(model)
    print(model)
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(clf["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    # get function handles of loss and metrics
    loss_module = instantiate(clf["loss"]).to(device)
    metrics_test = [
        instantiate(metric_dict)
        for metric_dict in clf["metrics"]
    ]
    metrics_train = metrics_test
    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(clf["optimizer"], trainable_params)
    scheduler = instantiate(clf["scheduler"], optimizer)
    trainer = Trainer(
        model,
        loss_module,
        metrics_train,
        metrics_test,
        optimizer=optimizer,
        config=clf,
        device=device,
        log_step=clf["trainer"].get("log_step", 100),
        dataloader=dataloader,
        scheduler=scheduler,
        len_epoch=clf["trainer"].get("len_epoch", None),
    )
    trainer.train()


if __name__ == "__main__":
    main()
