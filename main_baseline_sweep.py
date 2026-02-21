import pandas as pd
import torch
import itertools
import copy
from training.baseline_trainer import BaselineTrainer
from models.baseline_mlp import BaselineMLP
from utils.data_loader import load_data
from utils.logger import ExperimentLogger
import config
import os

# sweep config - running multiple experiments at once for MLP network
DATASETS_TO_RUN = ['california_housing'] #'slice_localization', 'higgs_100k', 'higgs_1M 
HIDDEN_LAYERS_LIST = [3]      
HIDDEN_DIMS_LIST = [32]     
RANDOM_SEEDS = [42]                  
def run_sweep():
    total_experiments = len(DATASETS_TO_RUN) * len(HIDDEN_LAYERS_LIST) * len(HIDDEN_DIMS_LIST) * len(RANDOM_SEEDS)
    current_exp = 0
    
    print(f"==================================================")
    print(f"STARTING BASELINE MLP SWEEP")
    print(f"Datasets: {DATASETS_TO_RUN}")
    print(f"Layers: {HIDDEN_LAYERS_LIST}")
    print(f"Dims: {HIDDEN_DIMS_LIST}")
    print(f"Total experiments: {total_experiments}")
    print(f"==================================================\n")

    for dataset_name in DATASETS_TO_RUN:
        
        # monkey patching
        if 'higgs' in dataset_name:
            config.BASE_DATASET_NAME = 'higgs'

            config.HIGGS_SIZE = dataset_name.replace('higgs_', '') # '100k', '1M'
            config.DATASET_NAME = dataset_name
            config.CURRENT_DATASET = config.DATASET_CONFIGS['higgs']
            config.SUBSET_SIZE = 100000 if config.HIGGS_SIZE == '100k' else 1000000
            config.USE_SUBSET = (config.HIGGS_SIZE != 'full')
            
            config.PROCESSED_DATA_DIR = os.path.join('data', 'processed', config.DATASET_NAME)
            
        else:
            config.BASE_DATASET_NAME = dataset_name
            config.DATASET_NAME = dataset_name
            config.CURRENT_DATASET = config.DATASET_CONFIGS[dataset_name]
            config.USE_SUBSET = False
            config.PROCESSED_DATA_DIR = os.path.join('data', 'processed', config.DATASET_NAME)
            
        print(f"\n---> Dataset: {config.DATASET_NAME}")
        
        try:
            train_loader, test_loader, input_dim, _ = load_data()
        except FileNotFoundError:
            print(f"     [SKIP] Data not found for {dataset_name}. Run preprocess first!")
            continue

        task_type = config.CURRENT_DATASET['task_type']

        # Grid Search loop
        for layers, hidden_dim, seed in itertools.product(HIDDEN_LAYERS_LIST, HIDDEN_DIMS_LIST, RANDOM_SEEDS):
            current_exp += 1
            print(f"   [{current_exp}/{total_experiments}] Model: {layers}x{hidden_dim} | Seed: {seed}")
            
            config.BASELINE_LAYERS = layers
            config.BASELINE_HIDDEN_DIM = hidden_dim
            config.RANDOM_SEED = seed
            
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                
            # Init Model
            model = BaselineMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=layers,
            ).to(config.DEVICE)
            
            # Init Trainer
            trainer = BaselineTrainer(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                task_type=task_type
            )
            
            # Run Training
            trainer.run_training()
            
            del model
            del trainer
            torch.cuda.empty_cache()

    print("\n==================================================")
    print("SWEEP COMPLETED!")
    print("Check logs/experiments.csv for results.")
    print("==================================================")

if __name__ == "__main__":
    run_sweep()
