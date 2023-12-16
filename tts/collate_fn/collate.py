import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from tts.collate_fn.functions import reprocess_tensor
from tts.datasets.MelSpectrogram import MelSpectrogram
from tts.utils.util import MelSpectrogramConfig


def collate_fn(dataset_items: List[dict]):
        """
        Collate and pad fields in dataset items
        """
        audio_length = torch.tensor(
            [dataset["audio"].shape[-1] for dataset in dataset_items]
        )
        target_audio = torch.vstack([dataset["audio"] for dataset in dataset_items])
        return {
            "audio": target_audio,
            "audio_length": audio_length,
            "speaker_id": [el["speaker_id"] for el in dataset_items],
            "algorithm": [el["algorithm"] for el in dataset_items],
            "is_real": torch.tensor([el["is_real"] for el in dataset_items]).long(),
        }
