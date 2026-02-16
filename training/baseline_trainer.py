import torch
import torch.nn as nn
import config
from utils.metrics import calculate_r2

def train_baseline(model, train_loader, test_loader):

    model.to(config.DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.BASELINE_LEARNING_RATE)  

    train_losses = []
    test_losses = []
    r2_scores = []

    print(f"Starting training Baseline model for ({config.BASELINE_EPOCHS}) epochs...")

    for epoch in range(config.BASELINE_EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Evaluate on test set
        test_loss = evaluate(model, test_loader, criterion)
        test_losses.append(test_loss)

        epoch_r2 = calculate_r2(model, test_loader, config.DEVICE)
        r2_scores.append(epoch_r2)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{config.BASELINE_EPOCHS}], Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, R2: {epoch_r2:.4f}')
        
    return train_losses, test_losses, r2_scores 

def evaluate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

    return running_loss / len(test_loader.dataset)