import os
import time
import copy
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, mean_squared_error

import config
from utils.logger import ExperimentLogger

# parent class for GrowNet trainer and BaselineMLP trainer
class BaseTrainer:
    def __init__(self, model, train_loader, test_loader, config_params,
                 experiment_name: str, task_type: str = 'regression'):
        
        self.model = model                  #nn.Module
        self.train_loader = train_loader    #DataLoader
        self.test_loader = test_loader      #DataLoader
        self.config = config_params         #dict {lr, epochs...}
        self.exp_name = experiment_name     #'GrowNet', 'BaselineMLP'
        self.task_type = task_type          #'regression', 'classification'
        self.device = config.DEVICE
        self.model.to(self.device)

        # loss and metrics depending on task type
        if self.task_type == 'regression':
            self.criterion = nn.MSELoss()
            self.metric_names = ['test_loss', 'rmse', 'r2']
            self.minimize_metric = True
            self.best_metric_val = float('inf')
            self.monitor_metric = 'test_loss'
        elif self.task_type == 'classification':
            self.criterion = nn.BCEWithLogitsLoss()
            self.metric_names = ['test_loss', 'accuracy', 'auc']
            self.minimize_metric = False
            self.best_metric_val = float('-inf')
            self.monitor_metric = 'auc'
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # logger and logging parameters
        self.logger = ExperimentLogger(
            model_type=experiment_name,
            dataset_name=getattr(config, 'DATASET_NAME', 'unknown_dataset'),
            params=config_params,
            metric_names=self.metric_names
        )
        self.timestamp = self.logger.timestamp
        self.total_training_time = 0.0

        # paths
        os.makedirs(config.SAVE_DIR_MODELS, exist_ok=True)
        os.makedirs(getattr(config, 'SAVE_DIR_PLOTS', 'plots'), exist_ok=True)

        # early stopping params
        self.patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 10)
        self.min_delta = getattr(config, 'EARLY_STOPPING_MIN_DELTA', 0.0)
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    # abstract method – child classes must implement train_epoch_step
    def train_epoch_step(self):
        raise NotImplementedError

    # evaluation
    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)

                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        epoch_loss = running_loss / len(self.test_loader.dataset)
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        metrics_dict = self._calculate_all_metrics(all_preds, all_targets)
        metrics_dict['test_loss'] = epoch_loss
        return metrics_dict

    def _calculate_all_metrics(self, preds, targets):
        metrics = {}
        if self.task_type == 'regression':
            mse = mean_squared_error(targets, preds)
            metrics['rmse'] = np.sqrt(mse)
            metrics['r2'] = r2_score(targets, preds)
        elif self.task_type == 'classification':
            probs = torch.sigmoid(torch.tensor(preds)).numpy()
            pred_labels = (probs > 0.5).astype(int)
            metrics['accuracy'] = accuracy_score(targets, pred_labels)
            try:
                metrics['auc'] = roc_auc_score(targets, probs)
            except ValueError:
                metrics['auc'] = 0.0
        return metrics

    # Early stopping with restoring best model parameters
    def check_early_stopping(self, current_val):
        if self.best_score is None:
            self.best_score = current_val
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            return False

        if self.minimize_metric:
            is_improvement = current_val < (self.best_score - self.min_delta)
        else:
            is_improvement = current_val > (self.best_score + self.min_delta)

        if is_improvement:
            self.best_score = current_val
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.counter = 0
            return False
        else:
            self.counter += 1
            print(f"    [EarlyStopping] Counter: {self.counter}/{self.patience} (Best: {self.best_score:.4f})")
            if self.counter >= self.patience:
                return True
            return False

    def restore_best_model(self):
        if self.best_model_state is not None:
            try:
                self.model.load_state_dict(self.best_model_state)
            except Exception as e:
                print(f"[WARN] Failed to restore best model: {e}")


    # main training loop
    def run_training(self):
        print(f"\n[INFO] Starting training: {self.exp_name} ({self.task_type})")
        print(f"[INFO] Logging to: logs/exp_{self.logger.exp_id}.csv")

        for epoch in range(self.config['epochs']):
            start_epoch = time.time()

            train_loss = self.train_epoch_step()
            metrics = self.evaluate()

            end_epoch = time.time()
            epoch_duration = end_epoch - start_epoch
            self.total_training_time += epoch_duration

            self.logger.log_step(epoch + 1, train_loss, metrics, epoch_duration)

            if (epoch + 1) % 5 == 0:
                log_str = f"Epoch [{epoch+1}/{self.config['epochs']}] "
                log_str += f"Train Loss: {train_loss:.4f} | "
                for key in self.metric_names:
                    val = metrics.get(key, 0.0)
                    log_str += f"{key.upper()}: {val:.4f} | "
                log_str += f"Time: {epoch_duration:.2f}s"
                print(log_str)

            current_metric = metrics[self.monitor_metric]
            if self.check_early_stopping(current_metric):
                print(f"\n[INFO] Early stopping triggered at epoch {epoch+1}!")
                print(f"[INFO] Restoring best model weights (Score: {self.best_score:.4f})")
                self.restore_best_model()
                break

        print(f"\n[INFO] Training finished in {self.total_training_time:.2f}s")

        if self.best_model_state is not None:
            self.restore_best_model()

        final_metrics = self.evaluate()
        self.logger.finish(train_loss, final_metrics)
        self.save_model()

    def save_model(self):
        model_filename = f"{self.exp_name}_{self.timestamp}.pth"
        model_path = os.path.join(config.SAVE_DIR_MODELS, model_filename)
        torch.save(self.model.state_dict(), model_path)
        print(f"[INFO] Model saved to {model_path}")
