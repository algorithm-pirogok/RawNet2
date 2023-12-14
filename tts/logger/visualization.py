from enum import Enum

from .tensorboard import TensorboardWriter
from .wandb import WanDBWriter


class VisualizerBackendType(str, Enum):
    wandb = "wandb"


def get_visualizer(config, logger, backend: VisualizerBackendType):
    if backend == VisualizerBackendType.wandb:
        return WanDBWriter(config, logger)
    else:
        raise ValueError("Please, use wandb")
