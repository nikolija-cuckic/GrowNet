import torch
import torch.nn as nn
import time
import os
import copy
import numpy as np
import config
from datetime import datetime
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, mean_squared_error
from utils.logger import ExperimentLogger  

class BaseTrainer:
    def __init__(self, model, train_loader, test_loader, config_params, experiment_name, task_type='regression'):
        """
        Args:
            model: PyTorch model
            train_loader: DataLoader za trening
            test_loader: DataLoader za test
            config_params: Dictionary sa parametrima (lr, epochs...)
            experiment_name: Ime eksperimenta (npr. 'BaselineMLP', 'GrowNet')
            task_type: 'regression' ili 'classification'
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config_params
        self.exp_name = experiment_name
        self.task_type = task_type
        self.device = config.DEVICE
        self.model.to(self.device)

        # 1. Definisanje Loss-a i Metrike na osnovu tipa zadatka
        if self.task_type == 'regression':
            self.criterion = nn.MSELoss()
            self.metric_names = ['test_loss', 'rmse', 'r2']  # Imena kolona za logger
            self.minimize_metric = True         # RMSE minimizujemo
            self.best_metric_val = float('inf')
            self.monitor_metric = 'test_loss' # metrics for early stopping
        elif self.task_type == 'classification':
            self.criterion = nn.BCEWithLogitsLoss()
            self.metric_names = ['test_loss', 'accuracy', 'auc']
            self.minimize_metric = False        # Accuracy maksimizujemo
            self.best_metric_val = float('-inf')
            self.monitor_metric = 'auc' # metrics for early stopping
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # 2. Inicijalizacija Loggera
        # Prosledjujemo imena metrika da logger zna sta da ocekuje
        self.logger = ExperimentLogger(
            model_type=experiment_name,
            dataset_name=getattr(config, 'DATASET_NAME', 'unknown_dataset'),
            params=config_params,
            metric_names=self.metric_names 
        )
        
        # Koristimo timestamp iz loggera da se svi fajlovi slazu
        self.timestamp = self.logger.timestamp
        self.total_training_time = 0.0
        
        # 3. Kreiranje foldera za modele i plotove
        os.makedirs(config.SAVE_DIR_MODELS, exist_ok=True)
        os.makedirs(getattr(config, 'SAVE_DIR_PLOTS', 'plots'), exist_ok=True)

        #early stopping
        self.patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 10)
        self.min_delta = getattr(config, 'EARLY_STOPPING_MIN_DELTA', 0.0)
        self.counter = 0 # Brojac koliko dugo nema poboljsanja
        self.best_score = None # Najbolja vrednost metrike do sad
        self.early_stop = False
        self.best_model_state = None # Ovde cemo cuvati najbolje tezine

    def train_epoch_step(self):
        """Mora biti implementirano u BaselineTrainer i GrowNetTrainer."""
        raise NotImplementedError

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

                # Skupljamo predikcije za metriku
                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        # Average Loss (MSE za regresiju, BCE za klasifikaciju)
        epoch_loss = running_loss / len(self.test_loader.dataset)
        
        # Concatenate sve batcheve
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        # Izracunaj sve metrike
        metrics_dict = self._calculate_all_metrics(all_preds, all_targets)
        
        # Dodaj test_loss u dict (uvek korisno)
        metrics_dict['test_loss'] = epoch_loss
        
        return metrics_dict

    def _calculate_all_metrics(self, preds, targets):
        """Računa sve metrike zavisno od task_type."""
        metrics = {}
        
        if self.task_type == 'regression':
            # RMSE
            mse = mean_squared_error(targets, preds)
            metrics['rmse'] = np.sqrt(mse)
            # R2 Score
            metrics['r2'] = r2_score(targets, preds)
            
        elif self.task_type == 'classification':
            # Pretpostavka: preds su logits
            probs = torch.sigmoid(torch.tensor(preds)).numpy()
            pred_labels = (probs > 0.5).astype(int)
            
            # Accuracy
            metrics['accuracy'] = accuracy_score(targets, pred_labels)
            # AUC (samo ako imamo 2 klase)
            try:
                metrics['auc'] = roc_auc_score(targets, probs)
            except ValueError:
                metrics['auc'] = 0.0 

        return metrics
    
    def check_early_stopping(self, current_val):
        """Proverava da li treba zaustaviti trening."""
        
        if self.best_score is None:
            self.best_score = current_val
            self.best_model_state = copy.deepcopy(self.model.state_dict()) # Sacuvaj prvu verziju
            return False

        # Logika: Da li je score bolji?
        if self.minimize_metric:
            # Ako minimizujemo (npr. Loss), "bolje" znaci MANJE
            is_improvement = current_val < (self.best_score - self.min_delta)
        else:
            # Ako maksimizujemo (npr. Accuracy), "bolje" znaci VECE
            is_improvement = current_val > (self.best_score + self.min_delta)

        if is_improvement:
            self.best_score = current_val
            self.best_model_state = copy.deepcopy(self.model.state_dict()) # Sacuvaj najbolje tezine
            self.counter = 0 # Resetuj brojac
            return False
        else:
            self.counter += 1
            print(f"    [EarlyStopping] Counter: {self.counter}/{self.patience} (Best: {self.best_score:.4f})")
            if self.counter >= self.patience:
                return True # STOP!
            return False   

    def restore_best_model(self):
        """Vraca najbolje tezine. Child klase (GrowNet) mogu ovo da override-uju."""
        if self.best_model_state is not None:
            # Default ponasanje za statičke modele (BaselineMLP)
            # Ovo ce se izvrsiti za Baseline, a za GrowNet ce se izvrsiti ONA TVOJA metoda
            try:
                self.model.load_state_dict(self.best_model_state)
            except Exception as e:
                print(f"[WARN] Failed to restore best model: {e}")


    def run_training(self):
        print(f"\n[INFO] Starting training: {self.exp_name} ({self.task_type})")
        print(f"[INFO] Logging to: logs/exp_{self.logger.exp_id}.csv")
        
        start_global = time.time()

        for epoch in range(self.config['epochs']):
            start_epoch = time.time()
            
            # --- 1. Train Step (Abstract) ---
            train_loss = self.train_epoch_step()
            
            # --- 2. Eval Step ---
            # Vraća rečnik sa svim metrikama (npr. {'test_loss': 0.5, 'rmse': 0.7, 'r2': 0.8})
            metrics = self.evaluate()
            
            # --- 3. Timing ---
            end_epoch = time.time()
            epoch_duration = end_epoch - start_epoch
            self.total_training_time += epoch_duration

            # --- 4. Log Step (CSV) ---
            # Prosledjujemo metrike loggeru. On ce izvuci sta mu treba (npr. rmse, r2)
            self.logger.log_step(epoch + 1, train_loss, metrics, epoch_duration)

            # --- 5. Print (Konzola) ---
            if (epoch + 1) % 5 == 0:
                log_str = f"Epoch [{epoch+1}/{self.config['epochs']}] "
                log_str += f"Train Loss: {train_loss:.4f} | "
                
                for key in self.metric_names:
                    val = metrics.get(key, 0.0)
                    log_str += f"{key.upper()}: {val:.4f} | "
                
                log_str += f"Time: {epoch_duration:.2f}s"
                print(log_str)

            # -------------------------------------------------------
            # NOVO: Poziv Early Stopping provere
            # -------------------------------------------------------
            # Pratimo 'test_loss' (najsigurnije za overfit)
            current_metric = metrics[self.monitor_metric] 
            
            if self.check_early_stopping(current_metric):
                print(f"\n[INFO] Early stopping triggered at epoch {epoch+1}!")
                print(f"[INFO] Restoring best model weights (Score: {self.best_score:.4f})")

                self.restore_best_model()

                break

        # --- Kraj treninga ---
        total_time_str = f"{self.total_training_time:.2f}s"
        print(f"\n[INFO] Training finished in {total_time_str}")
        
        # Ako smo izasli iz petlje (bilo zbog kraja ili early stop-a),
        # model sada ima NAJBOLJE tezine (jer smo ih vratili u break-u, 
        # ili su ostale poslednje ako nije bilo ES-a).
        
        # Ali Pazi: Ako NIJE okinuo early stop, moramo biti sigurni da je best_model_state sacuvan
        if self.best_model_state is not None:
            self.restore_best_model()

        # 6. Final Log & Save
        # Ponovo evaluiraj model jer su sada ucitane NAJBOLJE tezine, a ne poslednje
        final_metrics = self.evaluate()
        self.logger.finish(train_loss, final_metrics)
        self.save_model()

    def save_model(self):
        model_filename = f"{self.exp_name}_{self.timestamp}.pth"
        model_path = os.path.join(config.SAVE_DIR_MODELS, model_filename)
        torch.save(self.model.state_dict(), model_path)
        print(f"[INFO] Model saved to {model_path}")

