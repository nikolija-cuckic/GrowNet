import pandas as pd
import numpy as np
import time
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
import config
from utils.data_loader import load_data
from utils.logger import ExperimentLogger

def main():
    print(f"\n[XGBoost] Loading data for dataset: {config.DATASET_NAME}...")
    
    # load_data returns TensorDataloadere, XGB needs numpy/pandas/DMatrix
    # extracting data from loadera back to numpy
    train_loader, test_loader, input_dim, _ = load_data()
    
    print("[XGBoost] Converting DataLoaders to Numpy for XGBoost...")
    
    def extract_from_loader(loader):
        X_list, y_list = [], []
        for x_batch, y_batch in loader:
            X_list.append(x_batch.numpy())
            y_list.append(y_batch.numpy())
        return np.vstack(X_list), np.vstack(y_list)

    X_train, y_train = extract_from_loader(train_loader)
    X_test, y_test = extract_from_loader(test_loader)
    
    # flatten y (XGBoost needs (N,) not (N,1))
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    print(f"          Train shape: {X_train.shape}")
    print(f"          Test shape:  {X_test.shape}")


    task_type = config.CURRENT_DATASET['task_type']
    
    # params config
    xgb_params = {
        'n_estimators': 1024,      # equivalent to epochs/stages
        'learning_rate': 0.1,     # shrinkage
        'max_depth': 6,           # tree depth
        'random_state': config.RANDOM_SEED,
        'n_jobs': -1              # all cpu cores
    }

    # model and metrics
    if task_type == 'regression':
        print("[XGBoost] Task: REGRESSION (XGBRegressor)")
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            **xgb_params
        )
        metric_names = ['test_loss', 'rmse', 'r2']
        eval_metric = 'rmse'
    else:
        print("[XGBoost] Task: CLASSIFICATION (XGBClassifier)")
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc', # ili 'logloss'
            **xgb_params
        )
        metric_names = ['test_loss', 'accuracy', 'auc']

    logger = ExperimentLogger(
        model_type="XGBoost",
        dataset_name=config.DATASET_NAME,
        params=xgb_params,
        metric_names=metric_names
    )

    print(f"[XGBoost] Training started...")
    start_time = time.time()
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False 
    )
    
    train_time = time.time() - start_time
    print(f"[XGBoost] Training finished in {train_time:.2f}s")

    # evaluation
    if task_type == 'regression':
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, train_preds)
        test_mse = mean_squared_error(y_test, test_preds)
        
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, test_preds)
        
        metrics = {
            'test_loss': test_mse, 
            'rmse': test_rmse,
            'r2': test_r2
        }
        
    else: # Classification
        train_preds_prob = model.predict_proba(X_train)[:, 1]
        test_preds_prob = model.predict_proba(X_test)[:, 1]
        test_preds_label = (test_preds_prob > 0.5).astype(int)
        
        # Loss (LogLoss / Binary Cross Entropy)
        from sklearn.metrics import log_loss
        train_loss = log_loss(y_train, train_preds_prob)
        test_loss = log_loss(y_test, test_preds_prob)
        
        test_acc = accuracy_score(y_test, test_preds_label)
        test_auc = roc_auc_score(y_test, test_preds_prob)
        
        metrics = {
            'test_loss': test_loss,
            'accuracy': test_acc,
            'auc': test_auc
        }
        train_mse = train_loss 
   
    # logging only final result
    print(f"[XGBoost] Final Results:")
    for k, v in metrics.items():
        print(f"          {k.upper()}: {v:.4f}")

    logger.log_step(xgb_params['n_estimators'], train_mse, metrics, train_time)
    logger.finish(train_mse, metrics)
    
    # model.save_model(f"checkpoints/xgboost_{config.DATASET_NAME}.json")

if __name__ == "__main__":
    main()
