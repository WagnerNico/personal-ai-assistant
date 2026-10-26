"""A compact depthwise CNN trained from scratch for Airis wake detection."""

import torch
from torch import nn


class DepthwiseBlock(nn.Module):
    def __init__(self, channels, dilation, normalization):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=(dilation, 1),
                      dilation=(dilation, 1), groups=channels, bias=False),
            normalization(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            normalization(channels), nn.ReLU(),
        )

    def forward(self, inputs):
        return self.layers(inputs) + inputs


class WakeNetwork(nn.Module):
    """Learn temporal acoustic patterns without an external speech embedding model."""

    def __init__(self, channels=24, architecture="airis-dscnn-v2"):
        super().__init__()
        self.architecture = architecture
        normalization = (lambda count: nn.GroupNorm(4, count)) if architecture.endswith("v2") else nn.BatchNorm2d
        dilations = (1, 2, 4, 8, 16) if architecture.endswith("v2") else (1, 2, 4, 8)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, (9, 3), stride=2, padding=(4, 1), bias=False),
            normalization(channels), nn.ReLU(),
            *(DepthwiseBlock(channels, dilation, normalization) for dilation in dilations),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(channels * 2), nn.Linear(channels * 2, 64),
            nn.SiLU(), nn.Dropout(0.1), nn.Linear(64, 1),
        )

    def forward(self, inputs):
        encoded = self.encoder(inputs.unsqueeze(1))
        summary = torch.cat((encoded.mean(dim=(2, 3)), encoded.amax(dim=(2, 3))), dim=1)
        return self.classifier(summary).squeeze(-1)


class ProbabilityNetwork(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, inputs):
        return torch.sigmoid(self.network(inputs))
