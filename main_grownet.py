import os
from datetime import datetime
import torch
import matplotlib.pyplot as plt

import config
from utils.data_loader import load_data
from training.grownet_trainer import train_grownet

def main():
    print("Loading data...")
    train_loader, test_loader, input_dim, _ = load_data()
    print(f"Data loaded successfully. Input dimension: {input_dim}")

    model, train_losses, test_losses, r2_scores = train_grownet(
        train_loader, test_loader, input_dim
    )

    # --- ČUVANJE MODELA ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(config.SAVE_DIR_MODELS, exist_ok=True)
    model_path = os.path.join(config.SAVE_DIR_MODELS, f"grownet_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"GrowNet model saved to {model_path}")

    # --- GRAFICI ---
    os.makedirs(config.SAVE_DIR_PLOTS, exist_ok=True)

    # Loss po stage-u
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Stage")
    plt.ylabel("MSE Loss")
    plt.title("GrowNet Stage-wise Loss")
    plt.legend()
    plt.grid()
    loss_plot_path = os.path.join(config.SAVE_DIR_PLOTS, f"grownet_loss_{timestamp}.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"GrowNet loss plot saved to {loss_plot_path}")

    # R2 po stage-u
    plt.figure(figsize=(10, 5))
    plt.plot(r2_scores, label="Test R2", color="green")
    plt.xlabel("Stage")
    plt.ylabel("R2 Score")
    plt.title("GrowNet Stage-wise R2")
    plt.legend()
    plt.grid()
    r2_plot_path = os.path.join(config.SAVE_DIR_PLOTS, f"grownet_r2_{timestamp}.png")
    plt.savefig(r2_plot_path)
    plt.close()
    print(f"GrowNet R2 plot saved to {r2_plot_path}")
    print(f"Final GrowNet Test R2: {r2_scores[-1]:.4f}")

if __name__ == "__main__":
    main()
