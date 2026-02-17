import os

RANDOM_SEED = 42
TEST_SIZE = 0.2   
BATCH_SIZE = 64   

# dataset parameters
BASE_DATASET_NAME = 'slice_localization'  # 'california_housing', 'slice_localization', 'higgs'

# --- HIGGS SPECIFIC SETTINGS ---
# Opcije: '100k', '1M'
HIGGS_SIZE = '1M' 

# Logika za generisanje konacnog imena dataset-a
if BASE_DATASET_NAME == 'higgs':
    DATASET_NAME = f'higgs_{HIGGS_SIZE}' # npr. 'higgs_100k'
    
    # Mapiranje velicina u broj redova
    size_map = {
        '100k': 100000,
        '1M': 1000000,
    }
    SUBSET_SIZE = size_map.get(HIGGS_SIZE, 1000000)
    USE_SUBSET = True 
else:
    DATASET_NAME = BASE_DATASET_NAME
    USE_SUBSET = False # Za ostale datasete (housing) ne koristimo subset (mali su)

# Konfiguracija (Sada je mapirana po BAZNOM imenu)
DATASET_CONFIGS = {
    'california_housing': {
        'raw_file': 'housing.csv',
        'target_col': 'median_house_value',
        'drop_cols': ['ocean_proximity'],
        'task_type': 'regression',
        'has_header': True
    },
    'slice_localization': {
        'raw_file': 'slice_localization_data.csv',
        'target_col': 'reference',
        'drop_cols': ['patientId'],
        'task_type': 'regression',
        'has_header': True
    },
    'higgs': {
        'raw_file': 'HIGGS.csv',
        'target_col': 'class_label',
        'drop_cols': [],
        'task_type': 'classification',
        'has_header': False
    }
}

# Aktivni config (uvek gledamo BASE_DATASET_NAME jer raw fajl je isti)
CURRENT_DATASET = DATASET_CONFIGS[BASE_DATASET_NAME]

# Putanje
RAW_DATA_DIR = os.path.join('data', 'raw')
# Folder ce biti npr: data/processed/higgs_100k/
PROCESSED_DATA_DIR = os.path.join('data', 'processed', DATASET_NAME) 

# baseline MLP params
BASELINE_HIDDEN_DIM = 64
BASELINE_LAYERS = 2  # Broj skrivenih slojeva
BASELINE_LEARNING_RATE = 0.001
BASELINE_EPOCHS = 100

# GrowNet params
GROWNET_NUM_STAGES = 10 # broj weak learnera / boosting koraka
GROWNET_WEAK_HIDDEN_DIM = 64 # dim skrivenog sloja weak learnera 
GROWNET_WEAK_LR = 0.001 # learning rate za treniranje novog weak learnera
GROWNET_SHRINKAGE = 0.05 # boosting rate, doprinos svakog weak learnera

# corrective step
GROWNET_USE_CS = True # da li se primenjuje
GROWNET_CS_EPOCHS = 1 # koliko epoha za cs
GROWNET_CS_EVERY = 1 # na koliko boosting koraka

# early stopping criteriums
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.00001

# save directory for models and results
SAVE_DIR_MODELS = "checkpoints"
SAVE_DIR_PLOTS = "plots"

# Device configuration (GPU if available)
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
