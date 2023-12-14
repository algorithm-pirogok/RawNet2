import json
import os
from random import shuffle

import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

from tts.utils.util import ROOT_PATH


class BufferDataset(Dataset):
    def __init__(self, path_to_dataset: str, type_of_dataset: str, slice_length: int):
        self.slice_length = slice_length
        self.type = type_of_dataset
        path_to_json = ROOT_PATH / "data" / f"{self.type}.json"
        if not os.path.exists(path_to_json):
            self._create_json(path_to_dataset)
        with open(path_to_json, "r") as lst:
            self.dataset = json.load(lst)

    def _create_json(self, path_to_dataset):
        root = ROOT_PATH / path_to_dataset
        path_to_flac = root / f"ASVspoof2019_LA_{self.type}" / "flac"
        mode = "trn" if self.type == "train" else "trl"
        path_to_config = root / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{self.type}.{mode}.txt"
        path_to_json = ROOT_PATH / "data" / f"{self.type}.json"
        output_lst = []
        sh = 10000000
        with open(path_to_config, "r") as path_to_config:
            for line in tqdm(path_to_config, desc="Create config"):
                speaker_id, filename, _, algorithm, type_of_spoofed = line.split()
                path_to_file = path_to_flac / f"{filename}.flac"
                output_lst.append({
                    "speaker_id": speaker_id,
                    "path_to_file": str(path_to_file),
                    "algorithm": algorithm,
                    "is_spoofed": type_of_spoofed == 'spoof'
                })
        shuffle(output_lst)
        with open(path_to_json, 'w') as file:
            json.dump(output_lst, file, indent=4)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        elem = self.dataset[idx]
        audio, _ = torchaudio.load(elem['path_to_file'])
        audio = audio.squeeze()
        while audio.shape[-1] < self.slice_length:
            audio = audio.repeat(2)
        start_index = torch.randint(
            0, audio.shape[-1] - self.slice_length + 1, (1,)
        )
        audio = audio[start_index: start_index + self.slice_length]

        return {
            "speaker_id": elem["speaker_id"],
            "audio": audio,
            "algorithm": elem["algorithm"],
            "is_spoofed": elem["is_spoofed"]
            }
