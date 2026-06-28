import torch
import numpy as np
from sklearn.metrics import f1_score

def evaluate(model, loader, criterion, threshold=0.5):
    model.eval()
    device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu")
    total_loss = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, lengths, y in loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            logits = model(x, lengths)
            loss = criterion(logits, y)

            probs = torch.sigmoid(logits)

            total_loss += loss.item() * x.size(0)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    preds = (all_probs >= threshold).astype(int)
    f1 = f1_score(all_labels, preds)

    return total_loss / len(loader.dataset), f1, all_probs, all_labels