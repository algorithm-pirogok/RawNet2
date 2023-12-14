import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, weights: list[float]):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=torch.tensor(weights))

    def forward(self, pred_spoof, is_spoof):
        return self.cross_entropy_loss(pred_spoof, is_spoof)
