import torch
import torch.nn as nn

# a simple multi layer perceptron network, serving as a baseline for experiments with GrowNet
class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BaselineMLP, self).__init__()
        
        layers = []

        # input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # output layer 
        layers.append(nn.Linear(hidden_dim, 1))  
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)