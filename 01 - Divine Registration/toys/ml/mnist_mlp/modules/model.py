import torch.nn as nn
import torch.nn.functional as F


class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self._l1 = nn.Linear(784, 128)
        self._l2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self._l1(x)
        x = F.relu(x)
        x = self._l2(x)
        return x
