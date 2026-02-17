import pandas as pd
import torch
import itertools
import os
from training.grownet_trainer import GrowNetTrainer
from models.grownet import GrowNet
from utils.data_loader import load_data
import config

# --- KONFIGURACIJA SWEEP-A ---
DATASETS_TO_RUN = ['slice_localization'] #, 'higgs_100k']

# Parametri za testiranje (Grid Search)
NUM_STAGES_LIST = [10, 20, 30]
HIDDEN_DIMS_LIST = [32, 64]
SHRINKAGE_LIST = [0.05, 0.1]
USE_CS_LIST = [False]  # Mozes dodati False ako zelis ablaciju
CS_EVERY_LIST = [1]   # Na koliko koraka ide CS (ako je CS=True)
RANDOM_SEEDS = [42]

def run_sweep():
    # Kreiramo sve kombinacije parametara
    param_combinations = list(itertools.product(
        NUM_STAGES_LIST, 
        HIDDEN_DIMS_LIST, 
        SHRINKAGE_LIST, 
        USE_CS_LIST, 
        CS_EVERY_LIST,
        RANDOM_SEEDS
    ))
    
    total_experiments = len(DATASETS_TO_RUN) * len(param_combinations)
    current_exp = 0
    
    print(f"==================================================")
    print(f"STARTING GROWNET SWEEP")
    print(f"Datasets: {DATASETS_TO_RUN}")
    print(f"Total experiments: {total_experiments}")
    print(f"==================================================\n")

    for dataset_name in DATASETS_TO_RUN:
        
        # --- 1. Dataset Config Setup (Isto kao za Baseline) ---
        if 'higgs' in dataset_name:
            config.BASE_DATASET_NAME = 'higgs'
            config.HIGGS_SIZE = dataset_name.replace('higgs_', '')
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
        
        # --- 2. Ucitavanje podataka ---
        try:
            train_loader, test_loader, input_dim, _ = load_data()
        except FileNotFoundError:
            print(f"     [SKIP] Data not found for {dataset_name}. Run preprocess first!")
            continue

        task_type = config.CURRENT_DATASET['task_type']

        # --- 3. Grid Search Petlja ---
        for params in param_combinations:
            (stages, hidden_dim, shrinkage, use_cs, cs_every, seed) = params
            
            current_exp += 1
            print(f"   [{current_exp}/{total_experiments}] GrowNet: {stages} stages, {hidden_dim} dim, shr={shrinkage}, CS={use_cs} (every {cs_every})")
            
            # --- Postavljamo parametre u config ---
            config.GROWNET_NUM_STAGES = stages
            config.GROWNET_WEAK_HIDDEN_DIM = hidden_dim
            config.GROWNET_SHRINKAGE = shrinkage
            config.GROWNET_USE_CS = use_cs
            config.GROWNET_CS_EVERY = cs_every
            
            # Fiksni parametri (mozes i njih da sweep-ujes ako hoces)
            config.GROWNET_WEAK_LR = 0.001 
            config.GROWNET_CS_EPOCHS = 1
            config.RANDOM_SEED = seed
            
            # Set seed
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                
            # Init Model (GrowNet pocinje prazan ili sa 1. stage-om, zavisno od implementacije)
            model = GrowNet(
                input_dim=input_dim,
                output_dim=1,
                num_stages=stages,
                weak_hidden_dim=hidden_dim,
                task_type=task_type
            ).to(config.DEVICE)
            
            # Init Trainer
            trainer = GrowNetTrainer(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                task_type=task_type
            )
            
            # Run Training
            trainer.run_training()
            
            # Cleanup
            del model
            del trainer
            torch.cuda.empty_cache()

    print("\n==================================================")
    print("SWEEP COMPLETED!")
    print("Check logs/experiments.csv for results.")
    print("==================================================")

if __name__ == "__main__":
    run_sweep()
