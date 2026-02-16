import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import config
from models.grownet import GrowNet
from utils.metrics import calculate_r2
from utils.data_loader import load_data

def train_grownet(train_loader, test_loader, input_dim):
    device = config.DEVICE
    criterion = nn.MSELoss()

    model = GrowNet(input_dim).to(device)

    stage_train_losses = []
    stage_test_losses = []
    stage_r2_scores = []

    print(f"Starting GrowNet training with {config.GROWNET_NUM_STAGES} stages...")

    for stage in range(1, config.GROWNET_NUM_STAGES + 1):
        print(f"Stage {stage} / {config.GROWNET_NUM_STAGES}")

        new_wl = model.add_weak_learner().to(device)

        for p in model.parameters():
            p.requires_grad = False
        for p in new_wl.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(new_wl.parameters(), lr = config.GROWNET_WEAK_LR)

        #training new weak learner on residuals
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            with torch.no_grad():
                y_pred_prev = model(x_batch) - config.GROWNET_SHRINKAGE * new_wl(x_batch)  # without new wl
            residuals = y_batch - y_pred_prev

            pred_res = new_wl(x_batch)
            loss = criterion(pred_res, residuals)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)

        stage_train_loss = running_loss / len(train_loader.dataset)

        #corrective step, training all weak learners on true y
        if config.GROWNET_USE_CS and (stage % config.GROWNET_CS_EVERY == 0):
            for p in model.parameters():
                p.requires_grad = True
            cs_optimizer = torch.optim.Adam(model.parameters(), lr=config.GROWNET_WEAK_LR)

            for i in range(1, config.GROWNET_CS_EPOCHS+1):
                model.train()
                cs_running_loss = 0.0

                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)  

                    y_pred = model(x_batch)
                    cs_loss = criterion(y_pred, y_batch)

                    cs_optimizer.zero_grad()
                    cs_loss.backward()
                    cs_optimizer.step()

                    cs_running_loss += cs_loss.item() * x_batch.size(0)

                stage_train_loss = cs_running_loss / len(train_loader.dataset)


        stage_test_loss = evaluate_grownet(model, test_loader, criterion, device)
        r2 = calculate_r2(model, test_loader, device)

        stage_train_losses.append(stage_train_loss)
        stage_test_losses.append(stage_test_loss)   
        stage_r2_scores.append(r2)

        print(f"Stage {stage}: Train Loss={stage_train_loss:.4f}, Test Loss={stage_test_loss:.4f}, R2={r2:.4f}")

    return model, stage_train_losses, stage_test_losses, stage_r2_scores

def evaluate_grownet(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            running_loss += loss.item() * x_batch.size(0)
    return running_loss / len(loader.dataset)

            

