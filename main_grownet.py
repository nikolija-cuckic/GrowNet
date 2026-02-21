import config
from utils.data_loader import load_data
from models.grownet import GrowNet
from training.grownet_trainer import GrowNetTrainer

def main():
    # main grownet, loggs results and saves best model
    # all params set via config.py
    print("\n[GrowNet Main] Loading data...")
    
    train_loader, test_loader, input_dim, _ = load_data()
    
    print(f"[GrowNet Main] Data loaded successfully.")
    print(f"               Input dimension: {input_dim}")
    print(f"               Train samples: {len(train_loader.dataset)}")
    print(f"               Test samples:  {len(test_loader.dataset)}")


    print("\n[GrowNet Main] Initializing GrowNet model...")
    
    model = GrowNet(input_dim)
    
    model.to(config.DEVICE)
    print(f"[GrowNet Main] Model initialized on {config.DEVICE}")
    
    task_type = config.CURRENT_DATASET['task_type']

    print(f"[GrowNet Main] Starting training with {config.GROWNET_NUM_STAGES} stages...")
    
    trainer = GrowNetTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        task_type=task_type  
    )
    trainer.run_training()
    print("\n[GrowNet Main] Process completed successfully.")

if __name__ == "__main__":
    main()
