import matplotlib.pyplot as plt
import config
import os
import torch
from datetime import datetime
from utils.data_loader import load_data
from training.baseline_trainer import train_baseline
from models.baseline_mlp import BaselineMLP

def main():
    print("Loading data...")
    train_loader, test_loader, input_dim, _ = load_data()
    print(f"Data loaded successfully. Input dimension: {input_dim}")

    # Initialize model
    baseline_model = BaselineMLP(input_dim=input_dim, 
                        hidden_dim=config.BASELINE_HIDDEN_DIM, 
                        num_layers=config.BASELINE_LAYERS)

    # Train model
    train_losses, test_losses, r2_scores = train_baseline(baseline_model, train_loader, test_loader)


    # Save the trained model
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(config.SAVE_DIR_MODELS, exist_ok=True)
    model_filename = f'baseline_mlp_{timestamp}.pth'
    save_path = os.path.join(config.SAVE_DIR_MODELS, model_filename)
    torch.save(baseline_model.state_dict(), save_path)
    print(f"Baseline model saved to {save_path}")


    save_dir_plots = getattr(config, 'SAVE_DIR_PLOTS', 'plots') 
    os.makedirs(save_dir_plots, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Baseline MLP Loss') 
    plt.legend()
    plt.grid()
    
    plot_filename = f'loss_plot_{timestamp}.png'
    plot_path = os.path.join(save_dir_plots, plot_filename)
    
    plt.savefig(plot_path)  
    plt.close()             
    
    print(f"Loss plot saved to {plot_path}")

    # --- PLOT R2 SCORE ---
    plt.figure(figsize=(10, 5))
    plt.plot(r2_scores, label='Test R2 Score', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.title(f'Baseline MLP R2 Score')
    plt.legend()
    plt.grid()
    
    # Ime slike sa timestamp-om (isti timestamp kao za Loss da budu par)
    r2_plot_filename = f'r2_plot_{timestamp}.png'
    r2_plot_path = os.path.join(save_dir_plots, r2_plot_filename)
    
    plt.savefig(r2_plot_path)
    plt.close()
    
    print(f"R2 Score plot saved to {r2_plot_path}")
    print(f"Final Test R2 Score: {r2_scores[-1]:.4f}")


if __name__ == "__main__":
    main()

