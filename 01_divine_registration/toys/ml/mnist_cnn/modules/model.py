import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self._conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self._conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self._linear = nn.Linear(800, 10)

    def forward(self, x):
        # fmt: off
        x = self._conv1(x)                 # (bs, 16, 26, 26)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2) # (bs, 16, 13, 13)
        x = self._conv2(x)                 # (bs, 32, 11, 11)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2) # (bs, 32, 5, 5)
        x = torch.flatten(x, start_dim=1)  # (bs, 800)
        x = self._linear(x)                # (bs, 10)
        # fmt: on

        return x
