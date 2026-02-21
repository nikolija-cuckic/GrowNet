import torch
import torch.optim as optim
import config
from training.base_trainer import BaseTrainer

class BaselineTrainer(BaseTrainer):
    def __init__(self, model, train_loader, test_loader, task_type='regression'):

        # parameters for logging
        params = {
            'lr': config.BASELINE_LEARNING_RATE,
            'epochs': config.BASELINE_EPOCHS,
            'hidden_dim': config.BASELINE_HIDDEN_DIM,
            'layers': config.BASELINE_LAYERS
        }
        
        # initializing base trainer, 
        # creates Logger, sets loss (MSE/BCE) and metrics (RMSE/Acc)
        super().__init__(model, train_loader, test_loader, params, "BaselineMLP", task_type)
        
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.BASELINE_LEARNING_RATE)

    def train_epoch_step(self):
        self.model.train()
        running_loss = 0.0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward
            outputs = self.model(inputs)
            
            # Loss calculated using self.criterion in BaseTrainer
            # (MSE for regression, BCE for classification)
            loss = self.criterion(outputs, targets)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        # avg epoch loss for logging
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss
