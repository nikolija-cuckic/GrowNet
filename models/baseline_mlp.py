import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BaselineMLP, self).__init__()
        
        layers = []

        #Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        #Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer for regression
        layers.append(nn.Linear(hidden_dim, 1))  
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)