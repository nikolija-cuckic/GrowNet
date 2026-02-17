import config
from utils.data_loader import load_data
from models.grownet import GrowNet
from training.grownet_trainer import GrowNetTrainer

def main():
    # ---------------------------------------------------------
    # 1. Učitavanje Podataka
    # ---------------------------------------------------------
    print("\n[GrowNet Main] Loading data...")
    
    train_loader, test_loader, input_dim, _ = load_data()
    
    print(f"[GrowNet Main] Data loaded successfully.")
    print(f"               Input dimension: {input_dim}")
    print(f"               Train samples: {len(train_loader.dataset)}")
    print(f"               Test samples:  {len(test_loader.dataset)}")

    # ---------------------------------------------------------
    # 2. Inicijalizacija Modela (GrowNet)
    # ---------------------------------------------------------
    print("\n[GrowNet Main] Initializing GrowNet model...")
    
    # GrowNet se inicijalizuje kao PRAZAN kontejner (bez weak learner-a)
    # Prvi WL se dodaje automatski u prvoj epohi (stage-u) unutar Trainera
    model = GrowNet(input_dim)
    
    model.to(config.DEVICE)
    print(f"[GrowNet Main] Model initialized on {config.DEVICE}")

    # ---------------------------------------------------------
    # 3. Inicijalizacija Trenera (Boosting Logika)
    # ---------------------------------------------------------
    # Ovde biramo task_type ('regression' ili 'classification')
    
    task_type = config.CURRENT_DATASET['task_type']

    print(f"[GrowNet Main] Starting training with {config.GROWNET_NUM_STAGES} stages...")
    
    trainer = GrowNetTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        task_type=task_type  # Ključno: setuje MSE loss i RMSE metriku
    )

    # ---------------------------------------------------------
    # 4. Pokretanje Treninga (Stages)
    # ---------------------------------------------------------
    # Ova funkcija vrti petlju (Stages), dodaje WL-ove, radi CS, loguje i čuva
    trainer.run_training()

    print("\n[GrowNet Main] Process completed successfully.")

if __name__ == "__main__":
    main()
