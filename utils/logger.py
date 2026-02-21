import os
import csv
import json
import config
from datetime import datetime

class ExperimentLogger:
    def __init__(self, model_type, dataset_name, params, metric_names=['test_loss']):
        # metric_names: ['rmse', 'r2'] or ['accuracy', 'auc']
        
        self.model_type = model_type 
        self.dataset_name = dataset_name 
        self.params = params 
        self.metric_names = metric_names
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.logs_dir = getattr(config, 'LOGS_DIR', 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # main file with all experiments
        self.experiments_file = os.path.join(self.logs_dir, 'experiments.csv')
        
        # experiment ID
        self.exp_id = f"{model_type}_{dataset_name}_{self.timestamp}"
        
        # detailed log (by epochs/stages for baseline MLP/GrowNet)
        self.detail_log_file = os.path.join(self.logs_dir, f"exp_{self.exp_id}.csv")
        
        self.detail_headers = ['epoch', 'train_loss'] + metric_names + ['time']
        with open(self.detail_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.detail_headers)
            
        self.start_time = datetime.now()
        print(f"[Logger] Experiment started: {self.exp_id}")

    def log_step(self, epoch, train_loss, metrics_dict, epoch_time):
        # metrics_dict: (eg. {'rmse': 0.5, 'r2': 0.8 ...})
        row = [epoch, train_loss]
        # metrics defined in header
        for name in self.metric_names:
            row.append(metrics_dict.get(name, ''))
        row.append(epoch_time)
        
        with open(self.detail_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def finish(self, final_train_loss, final_metrics):
        duration = datetime.now() - self.start_time
        duration_str = str(duration).split('.')[0] 
        
        # experiment params and info
        experiment_data = {
            'exp_id': self.exp_id,
            'timestamp': self.timestamp,
            'model_type': self.model_type,
            'dataset': self.dataset_name,
            'duration': duration_str,
            'params': json.dumps(self.params),
            'final_train_loss': f"{final_train_loss:.6f}"
        }
        
        # adding final metrics for experiment.csv logging
        for name, val in final_metrics.items():
            experiment_data[f"final_{name}"] = f"{val:.6f}"
            
        # checking headers
        file_exists = os.path.isfile(self.experiments_file)
        
        fieldnames = ['exp_id', 'timestamp', 'model_type', 'dataset', 'duration', 
                      'final_train_loss', 'params']
        metric_cols = [f"final_{m}" for m in self.metric_names]
        fieldnames.extend(metric_cols)
        
        # append writing in experiments.csv         
        # file not exist - writing header
        with open(self.experiments_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            try:
                writer.writerow(experiment_data)
            except ValueError:
                print("[Logger Warning] CSV header mismatch. Appending ignoring extras.")

        print(f"[Logger] Experiment finished. Summary saved to {self.experiments_file}")
