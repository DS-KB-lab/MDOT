from torch import nn
import torch
from math import ceil


def osc_target_sampler(x: torch.Tensor, target_size: int) -> torch.Tensor:
    # TODO
    pass


class OSC(nn.Module):
    def __init__(self, channel_size=500, num_lead=12, output_channel=256):
        """
        :param channel_size: 2 * sample_rate
        :param num_lead: according to the dataset
        :param output_channel: fit the attention network
        """
        super(OSC, self).__init__()
        self.channel_size = channel_size
        self.num_lead = num_lead
        self.upsample = nn.Conv2d(self.num_lead, 128, kernel_size=1)
        self.enhance = nn.Conv2d(self.channel_size, self.channel_size, kernel_size=1)
        self.extraction = nn.Conv2d(128, output_channel, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Batch_size * Leads * Samples
        :return: Tokens for attention network
        """
        B, L, S = x.shape
        assert L == self.leads
        N = ceil(S / self.channel_size)
        x = nn.functional.pad(x, (0, N * self.channel_size - S))
        x = x.reshape(B, L, N, self.channel_size)
        x = self.upsample(x)
        x = self.enhance(x.transpose(1, 3))
        x = self.extraction(x.transpose(1, 3))
        x = torch.flatten(x.transpose(1, 2), start_dim=0, end_dim=1)
        return x
