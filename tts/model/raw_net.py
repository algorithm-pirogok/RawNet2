import torch
import torch.nn as nn
import torch.nn.functional as F

from tts.model import ResBlock, SincConv_fast


class RawNet(nn.Module):
    def __init__(self, sinc_config, resblock_config, gru_config):
        super().__init__()
        self.sinc_conv = SincConv_fast(out_channels=sinc_config.out_channels,
                                       kernel_size=sinc_config.kernel_size)
        self.max_pooling = nn.MaxPool1d(3)
        self.batch_norms = nn.Sequential(nn.BatchNorm1d(num_features=sinc_config.out_channels),
                                         nn.LeakyReLU())
        self.res_block = ResBlock(num_of_nets=resblock_config.num_of_nets,
                                  num_of_layers=resblock_config.num_of_layers,
                                  channels=[sinc_config.out_channels] + resblock_config.channels,
                                  first_norm=resblock_config.first_norm)
        self.gru_norm = None if gru_config.norm is False else nn.Sequential(
            nn.BatchNorm1d(num_features=resblock_config.channels[-1]),
            nn.LeakyReLU()
        )
        self.gru_block = nn.GRU(input_size=resblock_config.channels[-1],
                                hidden_size=gru_config.hidden_size,
                                num_layers=gru_config.num_layers,
                                batch_first=True)

        self.final_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(in_features=gru_config.hidden_size, out_features=2)
        )

    def forward(self, audio):
        ans = self.sinc_conv(audio.unsqueeze(1))
        ans = self.max_pooling(ans)
        ans = self.batch_norms(ans)
        ans = self.res_block(ans)
        ans = ans if self.gru_norm is None else self.gru_norm(ans)
        ans, _ = self.gru_block(ans.permute(0, 2, 1))
        return self.final_layer(ans.permute(1, 0, 2)[-1])
