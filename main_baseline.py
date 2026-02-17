import torch
import config
from utils.data_loader import load_data
from models.baseline_mlp import BaselineMLP
from training.baseline_trainer import BaselineTrainer

def main():
    # ---------------------------------------------------------
    # 1. Učitavanje Podataka
    # ---------------------------------------------------------
    print("\n[Main] Loading data...")
    
    # load_data vraća train_loader, test_loader, input_dim i scaler (koji nam ovde ne treba)
    train_loader, test_loader, input_dim, _ = load_data()
    
    print(f"[Main] Data loaded successfully.")
    print(f"       Input dimension: {input_dim}")
    print(f"       Train samples: {len(train_loader.dataset)}")
    print(f"       Test samples:  {len(test_loader.dataset)}")

    # ---------------------------------------------------------
    # 2. Inicijalizacija Modela (Baseline MLP)
    # ---------------------------------------------------------
    print("\n[Main] Initializing Baseline MLP model...")
    
    model = BaselineMLP(
        input_dim=input_dim, 
        hidden_dim=config.BASELINE_HIDDEN_DIM, 
        num_layers=config.BASELINE_LAYERS
    )
    
    # Prebacivanje na GPU/CPU (iako Trainer to radi opet, dobro je za proveru)
    model.to(config.DEVICE)
    print(f"[Main] Model initialized on {config.DEVICE}")

    # ---------------------------------------------------------
    # 3. Inicijalizacija Trenera
    # ---------------------------------------------------------
    # Ovde biramo task_type ('regression' ili 'classification')
    # Za California Housing koristimo 'regression'
    
    task_type = config.CURRENT_DATASET['task_type']

    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        task_type=task_type 
    )

    # ---------------------------------------------------------
    # 4. Pokretanje Treninga
    # ---------------------------------------------------------
    # Ova funkcija radi sve: vrti epohe, meri vreme, loguje u CSV i čuva model
    trainer.run_training()

    print("\n[Main] Process completed successfully.")

if __name__ == "__main__":
    main()
