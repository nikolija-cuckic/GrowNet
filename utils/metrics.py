from sklearn.metrics import r2_score
import torch

def calculate_r2(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            y_true.extend(targets.numpy())
            y_pred.extend(outputs.cpu().numpy())
            
    return r2_score(y_true, y_pred)
