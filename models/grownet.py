import torch
import torch.nn as nn
import config
from .weak_learner import WeakLearner

class GrowNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.models = nn.ModuleList()
        self.shrinkage = config.GROWNET_SHRINKAGE

    def add_weak_learner(self):
        wl = WeakLearner(self.input_dim, config.GROWNET_WEAK_HIDDEN_DIM)
        self.models.append(wl)
        return wl
    
    def forward(self, x):
        if len(self.models) == 0:
            return torch.zeros(x.size(0), 1, device = x.device)
        out = 0.0
        for m in self.models:
            out = out + self.shrinkage * m(x)
        return out

