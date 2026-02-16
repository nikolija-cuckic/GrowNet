# Dataset Parameters
DATASET_NAME = "california_housing"
TEST_SIZE = 0.2
RANDOM_SEED = 42
BATCH_SIZE = 64

# Baseline MLP Parameters
BASELINE_HIDDEN_DIM = 64
BASELINE_LAYERS = 2  # Broj skrivenih slojeva
BASELINE_LEARNING_RATE = 0.001
BASELINE_EPOCHS = 100

# GrowNet params
GROWNET_NUM_STAGES = 20 # broj weak learnera / boosting koraka
GROWNET_WEAK_HIDDEN_DIM = 64 # dim skrivenog sloja weak learnera 
GROWNET_WEAK_LR = 0.001 # learning rate za treniranje novog weak learnera
GROWNET_SHRINKAGE = 0.05 # boosting rate, doprinos svakog weak learnera

# corrective step
GROWNET_USE_CS = True # da li se primenjuje
GROWNET_CS_EPOCHS = 1 # koliko epoha za cs
GROWNET_CS_EVERY = 1 # na koliko boosting koraka

# Save directory for models and results
SAVE_DIR_MODELS = "checkpoints"
SAVE_DIR_PLOTS = "plots"

# Device configuration (GPU if available)
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
