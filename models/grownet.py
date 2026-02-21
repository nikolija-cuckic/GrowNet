import torch
import torch.nn as nn
import config
from .weak_learner import WeakLearner


class GrowNet(nn.Module):
    # GrowNet model: list of weak learners, trained with boosting
    # every next stage/wl has input: [x, penultimate features of previous wl].
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = config.GROWNET_WEAK_HIDDEN_DIM
        self.shrinkage = config.GROWNET_SHRINKAGE

        self.models = nn.ModuleList()

    def add_weak_learner(self):
        #first stage: input dim = original input dim
        #next stages: input dim = original input dim + hidden dim
       
        if len(self.models) == 0:
            current_input_dim = self.input_dim
        else:
            current_input_dim = self.input_dim + self.hidden_dim

        wl = WeakLearner(current_input_dim, self.hidden_dim)
        self.models.append(wl)
        return wl


    def forward_and_features(self, x, upto: int | None = None):
        # forward for 'upto' weak learners (None = for all wl)
        # returns out - final prediction, feats - penultimate features of last stage
        if len(self.models) == 0:
            zeros_out = torch.zeros(x.size(0), 1, device=x.device)
            zeros_feat = None
            return zeros_out, zeros_feat

        if upto is None:
            upto = len(self.models)

        out = None
        features = None

        # first stage inputs: x
        out_stage, features = self.models[0](x)
        out = self.shrinkage * out_stage

        # next stages inputs: [x, penultimate features of previous wl]
        for m in self.models[1:upto]:
            x_aug = torch.cat([x, features.detach()], dim=1)
            out_stage, features = m(x_aug)
            out = out + self.shrinkage * out_stage

        return out, features

    def forward(self, x):
        out, _ = self.forward_and_features(x, upto=None)
        return out
