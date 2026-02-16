import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import config
def preprocess():
#    raw_path = os.path.join('data', 'raw', 'housing.csv')
    raw_path = os.path.join('data', 'raw', 'slice_localization_data.csv')
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    print(f"Loading raw files {raw_path}")
    df = pd.read_csv(raw_path)

#    if "ocean_proximity" in df.columns:
#        df = df.drop('ocean_proximity', axis=1)

    if "patientId" in df.columns:
        df = df.drop('patientId', axis=1)
    
    df = df.dropna()

#    target_col = 'median_house_value'
    target_col = 'reference'
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.fit_transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1))
    y_test_scaled = scaler_y.fit_transform(y_test.values.reshape(-1,1))

    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df[target_col] = y_train_scaled

    test_df = pd.DataFrame(X_test_scaled, columns = X.columns)
    test_df[target_col] = y_test_scaled

    train_path = os.path.join(processed_dir, 'train.csv')
    test_path = os.path.join(processed_dir, 'test.csv')
    scaler_path = os.path.join(processed_dir, 'scalers.pkl')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    with open(scaler_path, 'wb') as f:
        pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)
    
    print(f"Data saved in {processed_dir}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

if __name__ == "__main__":
    preprocess()
 