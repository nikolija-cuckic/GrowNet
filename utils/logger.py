import os
import csv
import json
from datetime import datetime
import pandas as pd
import config

class ExperimentLogger:
    def __init__(self, model_type, dataset_name, params):
        self.model_type = model_type #grownet / baseline / xgboost
        self.dataset_name = dataset_name #cal housing, slice localization, higgs
        self.params = params  # all params for this experiment (e.g. {'lr': 0.01, 'layers': 2...})
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.logs_dir = getattr(config, 'LOGS_DIR', 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.experiments_file = os.path.join(self.logs_dir, 'experiments.csv')
        
        # generate experiment id
        self.exp_id = f"{model_type}_{dataset_name}_{self.timestamp}"
        
        self.detail_log_file = os.path.join(self.logs_dir, f"exp_{self.exp_id}.csv")
        
        with open(self.detail_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch_or_stage', 'train_loss', 'test_loss', 'r2_score', 'rmse'])
            
        self.start_time = datetime.now()
        print(f"Experiment started: {self.exp_id}")

    def log_step(self, epoch, train_loss, test_loss, r2, rmse=None):
        with open(self.detail_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, test_loss, r2, rmse if rmse else ''])

    def finish(self, final_train_loss, final_test_loss, final_r2, final_rmse=None):
        duration = datetime.now() - self.start_time
        duration_str = str(duration).split('.')[0]  # HH:MM:SS
        
        experiment_data = {
            'exp_id': self.exp_id,
            'timestamp': self.timestamp,
            'model_type': self.model_type,
            'dataset': self.dataset_name,
            'duration': duration_str,
            'final_train_loss': f"{final_train_loss:.6f}",
            'final_test_loss': f"{final_test_loss:.6f}",
            'final_r2': f"{final_r2:.6f}",
            'final_rmse': f"{final_rmse:.6f}" if final_rmse else "",
            'params': json.dumps(self.params)  # params as json
        }
        
        file_exists = os.path.isfile(self.experiments_file)
        
        fieldnames = ['exp_id', 'timestamp', 'model_type', 'dataset', 'duration', 
                      'final_train_loss', 'final_test_loss', 'final_r2', 'final_rmse', 'params']
        
        with open(self.experiments_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(experiment_data)
            
        print(f"Experiment finished. Summary saved to {self.experiments_file}")
