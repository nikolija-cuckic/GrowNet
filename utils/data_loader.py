import pandas as pd
import torch
import os
import pickle
from torch.utils.data import TensorDataset, DataLoader
import config

def load_data():
    # path in config (with DATASET_NAME)
    processed_dir = config.PROCESSED_DATA_DIR 
    
    train_path = os.path.join(processed_dir, 'train.csv')
    test_path = os.path.join(processed_dir, 'test.csv')
    scaler_path = os.path.join(processed_dir, 'scalers.pkl')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Data not found in {processed_dir}.\n"
            f"Please run: python scripts/preprocess_data.py"
        )
        
    print(f"[DataLoader] Loading data from {processed_dir}...")
        
    # loading csv
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # transforming X and y
    target_col = config.CURRENT_DATASET['target_col']
    
    X_train = train_df.drop(columns=[target_col]).values
    y_train = train_df[target_col].values.reshape(-1, 1)
    
    X_test = test_df.drop(columns=[target_col]).values
    y_test = test_df[target_col].values.reshape(-1, 1)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor), 
        batch_size=config.BATCH_SIZE, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor), 
        batch_size=config.BATCH_SIZE, 
        shuffle=False
    )
    
    # loading scaler for inverse transformations
    scaler_y = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
            scaler_y = scalers.get('scaler_y') 
            
    input_dim = X_train.shape[1]
    
    return train_loader, test_loader, input_dim, scaler_y
