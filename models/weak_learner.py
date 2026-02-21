import torch.nn as nn
import torch

# shallow mlp with 2 hidden layers (best results according to [1] Badirli et al. 2020)
# returns output + penultimate features to next weak learner input
# batchNorm and leakyReLU were used in [1], but here gave worse performance 
class WeakLearner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),  
            #nn.LeakyReLU(0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.regressor(features)
        return output, features
