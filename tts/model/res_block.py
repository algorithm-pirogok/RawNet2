import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self,
                 num_of_nets: int,
                 num_of_layers: list[int],
                 channels: list[int],
                 first_norm: bool):
        super().__init__()
        self.model = nn.ModuleList([nn.Sequential(*[ResNet(channels[net] if i == 0 else channels[net+1],
                                                           channels[net+1],
                                                           first_norm)
                                                    for i in range(num_of_layers[net])]) for net in range(num_of_nets)])

    def forward(self, x):
        for model in self.model:
            x = model(x)
        return x


class ResNet(nn.Module):
    def __init__(self, input_channels: int, convolution_channels: int, first_norm: bool):
        super().__init__()
        self.batch_norm = nn.Sequential(nn.BatchNorm1d(input_channels),
                                        nn.LeakyReLU()) if first_norm else None
        self.convolutions = nn.Sequential(
            nn.Conv1d(in_channels=input_channels,
                      out_channels=convolution_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.BatchNorm1d(num_features=convolution_channels),
            nn.Conv1d(in_channels=convolution_channels,
                      out_channels=convolution_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.LeakyReLU(),
        )
        self.down_sample = nn.Conv1d(
            in_channels=input_channels,
            out_channels=convolution_channels,
            padding=0,
            kernel_size=1,
            stride=1) if input_channels != convolution_channels else None
        self.median_pulling = nn.MaxPool1d(3)
        self.second_pulling = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc_layer = nn.Linear(in_features=convolution_channels, out_features=convolution_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ans = x if self.batch_norm(x) is None else self.batch_norm(x)
        ans = self.convolutions(ans)
        x = x if self.down_sample is None else self.down_sample(x)
        ans = self.median_pulling(ans + x)
        adding = self.sigmoid(self.fc_layer(self.second_pulling(ans).view(x.shape[0], -1))).unsqueeze(-1)
        return (ans + 1) * adding
