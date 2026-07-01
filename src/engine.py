import numpy as np
import torch
from sklearn.metrics import f1_score


def train_one_epoch_bert(model, loader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * batch["input_ids"].size(0)

    return total_loss / len(loader.dataset)


def evaluate_bert(model, loader, device, threshold=0.5):
    model.eval()

    total_loss = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            probs = torch.softmax(logits, dim=1)[:, 1]

            total_loss += loss.item() * batch["input_ids"].size(0)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    preds = (all_probs >= threshold).astype(int)
    f1 = f1_score(all_labels, preds)

    return total_loss / len(loader.dataset), f1, all_probs, all_labels


def predict_bert(model, loader, device):
    model.eval()

    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)