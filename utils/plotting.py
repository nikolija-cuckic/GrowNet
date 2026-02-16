import os
import matplotlib.pyplot as plt

def plot_metric(data_dict, metric_name, model_name, save_dir, filename_suffix=""):
    """
        data_dict (dict): dict with data for plotting, key = label
        metric_name (str): MSE Loss, R2 Score...
        model_name (str): 'GrowNet', 'MLP', 'XGBoost'
        filename_suffix (str): timestamp or experiment id
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    
    for label, values in data_dict.items():
        plt.plot(values, label=label)
        
    xlabel = 'Stage' if 'GrowNet' in model_name else 'Epoch'
    
    plt.xlabel(xlabel)
    plt.ylabel(metric_name)
    plt.title(f'{model_name} - {metric_name}')
    plt.legend()
    plt.grid(True)
    
    #model_metric_timestamp.png
    safe_metric_name = metric_name.lower().replace(" ", "_")
    filename = f"{model_name.lower().replace(' ', '_')}_{safe_metric_name}_{filename_suffix}.png"
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved: {save_path}")


def plot_loss(train_losses, test_losses, model_name, save_dir, timestamp):
    plot_metric(
        data_dict={'Train Loss': train_losses, 'Test Loss': test_losses},
        metric_name='MSE Loss',
        model_name=model_name,
        save_dir=save_dir,
        filename_suffix=timestamp
    )

def plot_r2(r2_scores, model_name, save_dir, timestamp):
    plot_metric(
        data_dict={'Test R2 Score': r2_scores},
        metric_name='R2 Score',
        model_name=model_name,
        save_dir=save_dir,
        filename_suffix=timestamp
    )

def plot_rmse(rmse_scores, model_name, save_dir, timestamp):
    plot_metric(
        data_dict={'Test RMSE': rmse_scores},
        metric_name='RMSE',
        model_name=model_name,
        save_dir=save_dir,
        filename_suffix=timestamp
    )
