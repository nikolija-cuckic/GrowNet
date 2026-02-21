import torch
import config
from utils.data_loader import load_data
from models.baseline_mlp import BaselineMLP
from training.baseline_trainer import BaselineTrainer

def main():
    # main, trains MLP model, loggs and saves best model
    # all params set via config.py
    print("\n[Main] Loading data...")
    train_loader, test_loader, input_dim, _ = load_data()
    
    print(f"[Main] Data loaded successfully.")
    print(f"       Input dimension: {input_dim}")
    print(f"       Train samples: {len(train_loader.dataset)}")
    print(f"       Test samples:  {len(test_loader.dataset)}")

    print("\n[Main] Initializing Baseline MLP model...")
    
    model = BaselineMLP(
        input_dim=input_dim, 
        hidden_dim=config.BASELINE_HIDDEN_DIM, 
        num_layers=config.BASELINE_LAYERS
    )
    
    model.to(config.DEVICE)
    print(f"[Main] Model initialized on {config.DEVICE}")
    
    task_type = config.CURRENT_DATASET['task_type']

    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        task_type=task_type 
    )
    trainer.run_training()

    print("\n[Main] Process completed successfully.")

if __name__ == "__main__":
    main()
