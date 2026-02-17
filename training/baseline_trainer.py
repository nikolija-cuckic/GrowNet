import torch
import torch.optim as optim
import config
from training.base_trainer import BaseTrainer

class BaselineTrainer(BaseTrainer):
    def __init__(self, model, train_loader, test_loader, task_type='regression'):
        """
        Args:
            model: PyTorch model (BaselineMLP)
            train_loader: DataLoader za trening
            test_loader: DataLoader za test
            task_type: 'regression' ili 'classification' (default: 'regression')
        """
        # 1. Priprema parametara za log
        params = {
            'lr': config.BASELINE_LEARNING_RATE,
            'epochs': config.BASELINE_EPOCHS,
            'hidden_dim': config.BASELINE_HIDDEN_DIM,
            'layers': config.BASELINE_LAYERS
        }
        
        # 2. Inicijalizacija BaseTrainer-a
        # On ce kreirati Logger, postaviti Loss (MSE/BCE) i Metrike (RMSE/Acc)
        super().__init__(model, train_loader, test_loader, params, "BaselineMLP", task_type)
        
        # 3. Optimizer (specifičan za model, zato je ovde)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.BASELINE_LEARNING_RATE)

    def train_epoch_step(self):
        """
        Implementira jedan korak treninga (forward + backward) za celu epohu.
        BaseTrainer ovo poziva u petlji.
        """
        self.model.train()
        running_loss = 0.0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward
            outputs = self.model(inputs)
            
            # Loss računamo koristeći self.criterion koji je setovan u BaseTrainer
            # (MSE za regresiju, BCE za klasifikaciju)
            loss = self.criterion(outputs, targets)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        # Vraćamo prosečan loss epohe (ovo ide u logove)
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss
