import torch
import torch.nn as nn


class HiFiLoss(nn.Module):
    def __init__(self, mel_coeff: float = 45, feature_coeff: float = 2):
        super().__init__()
        self.Loss = nn.L1Loss()
        self.mel_coeff = mel_coeff
        self.feature_coeff = feature_coeff

    def mel_loss(self, target_mels, pred_mels):
        return self.mel_coeff * self.Loss(target_mels, pred_mels)

    @staticmethod
    def discriminator_loss(target_prob, pred_prob):
        loss = 0
        for tg, pr in zip(target_prob, pred_prob):
            loss = loss + torch.mean((tg - 1) ** 2) + torch.mean(pr**2)
        return loss

    @staticmethod
    def generator_loss(pred_prob):
        loss = 0
        for pred in pred_prob:
            loss = loss + torch.mean((pred - 1) ** 2)
        return loss

    def feature_loss(self, target_prob, pred_prob):
        loss = 0
        for target, prob in zip(target_prob, pred_prob):
            for tg, pr in zip(target, prob):
                loss = loss + torch.mean(torch.abs(tg - pr))
        return self.feature_coeff * loss

    def forward(self, x):
        raise ValueError("Нет общего лосса")
