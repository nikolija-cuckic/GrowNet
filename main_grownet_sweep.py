import pandas as pd
import torch
import itertools
import os
from training.grownet_trainer import GrowNetTrainer
from models.grownet import GrowNet
from utils.data_loader import load_data
import config

# sweep config - running multiple experiments at once for MLP network
DATASETS_TO_RUN = ['california_housing'] #'slice_localization', 'higgs_100k', 'higgs_1M 

NUM_STAGES_LIST = [30]
HIDDEN_DIMS_LIST = [16]
SHRINKAGE_LIST = [0.1]
USE_CS_LIST = [True]
CS_EVERY_LIST = [1]
CS_EPOCHS_LIST = [1]
RANDOM_SEEDS = [42]
BATCH_SIZE_LIST = [1024] 


def run_sweep():
    param_combinations = list(itertools.product(
        NUM_STAGES_LIST,
        HIDDEN_DIMS_LIST,
        SHRINKAGE_LIST,
        USE_CS_LIST,
        CS_EVERY_LIST,
        CS_EPOCHS_LIST,
        RANDOM_SEEDS,
        BATCH_SIZE_LIST  
    ))

    total_experiments = len(DATASETS_TO_RUN) * len(param_combinations)
    current_exp = 0

    print(f"==================================================")
    print(f"STARTING GROWNET SWEEP")
    print(f"Datasets: {DATASETS_TO_RUN}")
    print(f"Total experiments: {total_experiments}")
    print(f"==================================================\n")

    for dataset_name in DATASETS_TO_RUN:
        # monkey patching over config
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

        task_type = config.CURRENT_DATASET['task_type']

        for params in param_combinations:
            (stages, hidden_dim, shrinkage, use_cs, cs_every, cs_epochs, seed, batch_size) = params

            current_exp += 1
            print(f"   [{current_exp}/{total_experiments}] GrowNet: {stages} stages, "
                  f"{hidden_dim} dim, shr={shrinkage}, CS={use_cs} (every {cs_every}), "
                  f"batch={batch_size}")   
            
            config.GROWNET_NUM_STAGES = stages
            config.GROWNET_WEAK_HIDDEN_DIM = hidden_dim
            config.GROWNET_SHRINKAGE = shrinkage
            config.GROWNET_USE_CS = use_cs
            config.GROWNET_CS_EVERY = cs_every
            config.GROWNET_CS_EPOCHS = cs_epochs
            config.GROWNET_WEAK_LR = 0.001
            config.RANDOM_SEED = seed
            config.BATCH_SIZE = batch_size   

            try:
                train_loader, test_loader, input_dim, _ = load_data()
            except FileNotFoundError:
                print(f"     [SKIP] Data not found for {dataset_name}.")
                continue

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            model = GrowNet(input_dim=input_dim).to(config.DEVICE)

            trainer = GrowNetTrainer(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                task_type=task_type
            )

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
