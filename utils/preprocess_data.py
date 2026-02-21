import pandas as pd
import os
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
# run only once on raw data, 
# get train.csv test.csv and scaler.pkl in separate folders for different datasets
def preprocess():
    dataset_name = config.DATASET_NAME
    ds_config = config.CURRENT_DATASET
    
    raw_path = os.path.join(config.RAW_DATA_DIR, ds_config['raw_file'])
    save_dir = config.PROCESSED_DATA_DIR
    
    print(f"\n[Preprocess] Processing dataset: {dataset_name}")
    print(f"             Raw file: {raw_path}")
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")
        
    os.makedirs(save_dir, exist_ok=True)
    
    load_params = {}
    
    # subset used for Higgs dataset (original - 11M rows, we use either 100k or 1M)
    if config.USE_SUBSET:
        print(f"             SUBSET ACTIVE: Reading first {config.SUBSET_SIZE} rows")
        load_params['nrows'] = config.SUBSET_SIZE
    
    # header
    if not ds_config.get('has_header', True):
        load_params['header'] = None

    print("             Loading CSV...")
    df = pd.read_csv(raw_path, **load_params)
    
    # if not header, assigning column names 
    if not ds_config.get('has_header', True):
        print("             Assigning column names...")
        # checking dimensions
        num_cols = df.shape[1]
        col_names = [ds_config['target_col']] + [f"feature_{i}" for i in range(num_cols - 1)]
        df.columns = col_names

    print(f"             Original shape: {df.shape}")

    # dropping columns (not number type and nulls)
    if ds_config['drop_cols']:
        df = df.drop(columns=ds_config['drop_cols'], errors='ignore')
        
    df = df.dropna()
    
    # split X / y
    target_col = ds_config['target_col']
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # train / test split, shuffle = true by default
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )

    # scaling via standard scaler both X and y for regression, only X for classification
    print("             Scaling...")
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = None
    if ds_config['task_type'] == 'regression':
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    else:
        y_train_scaled = y_train.values.reshape(-1, 1)
        y_test_scaled = y_test.values.reshape(-1, 1)

    # saving
    print("             Saving...")
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df[target_col] = y_train_scaled

    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df[target_col] = y_test_scaled

    train_path = os.path.join(save_dir, 'train.csv')
    test_path = os.path.join(save_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    with open(os.path.join(save_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)

    print(f"\n[Preprocess] Done! Saved to: {save_dir}")

if __name__ == "__main__":
    preprocess()
