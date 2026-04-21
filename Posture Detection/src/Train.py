import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    X_TRAIN_FILE, Y_TRAIN_FILE, CLASSIFIER_MODEL,
    MODELS_DIR, NUM_CLASSES, FEATURE_SIZE,
    EPOCHS, BATCH_SIZE, LEARNING_RATE,
    TRAIN_SPLIT, DROPOUT_RATE, LABEL_NAMES,
)

HISTORY_FILE = os.path.join(MODELS_DIR, "training_history.json")



class PostureClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.network(x)


def train():

    for path in [X_TRAIN_FILE, Y_TRAIN_FILE]:
        if not os.path.exists(path):
            print(f"\n[ERROR] File not found: {path}")
            print("        Run step4_build_dataset.py first.\n")
            sys.exit(1)

    os.makedirs(MODELS_DIR, exist_ok=True)

    X = torch.tensor(np.load(X_TRAIN_FILE), dtype=torch.float32)
    y = torch.tensor(np.load(Y_TRAIN_FILE), dtype=torch.long)

    print(f"\n[INFO] Dataset: {len(X)} samples, "
          f"{FEATURE_SIZE} features, {NUM_CLASSES} classes")

    dataset  = TensorDataset(X, y)
    n_train  = int(TRAIN_SPLIT * len(dataset))
    n_val    = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    print(f"[INFO] Train: {n_train}  |  Validation: {n_val}\n")

    model     = PostureClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss":   [],
        "val_accuracy": [],
        "epochs":       [],
    }

    best_val_acc = 0.0
    best_epoch   = 0

    print(f"  {'Epoch':>6}  {'Loss':>10}  {'Val Acc':>10}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}")

    for epoch in range(1, EPOCHS + 1):

        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                correct += (model(xb).argmax(1) == yb).sum().item()

        val_acc = correct / n_val

        history["epochs"].append(epoch)
        history["train_loss"].append(round(avg_loss, 6))
        history["val_accuracy"].append(round(val_acc, 6))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), CLASSIFIER_MODEL)

        if epoch % 5 == 0 or epoch == 1:
            marker = "  ◄ best" if epoch == best_epoch else ""
            print(f"  {epoch:>6}  {avg_loss:>10.4f}  {val_acc:>9.2%}{marker}")

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[INFO] Training history saved to: {HISTORY_FILE}")

    print(f"\n[DONE] Best epoch: {best_epoch}  |  "
          f"Best val accuracy: {best_val_acc:.2%}")
    print(f"[DONE] Model saved to: {CLASSIFIER_MODEL}\n")

    print("  Per-class accuracy on validation set:")
    model.load_state_dict(torch.load(CLASSIFIER_MODEL, map_location="cpu"))
    model.eval()

    class_correct = {i: 0 for i in range(NUM_CLASSES)}
    class_total   = {i: 0 for i in range(NUM_CLASSES)}
    all_preds     = []
    all_trues     = []

    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb).argmax(1)
            all_preds.extend(preds.tolist())
            all_trues.extend(yb.tolist())
            for pred, true in zip(preds, yb):
                class_total[true.item()]   += 1
                if pred.item() == true.item():
                    class_correct[true.item()] += 1

    per_class_acc = {}
    for cls in range(NUM_CLASSES):
        total = class_total[cls]
        if total == 0:
            acc = 0.0
            print(f"    {LABEL_NAMES[cls]:<15} : no samples")
        else:
            acc = class_correct[cls] / total
            bar = "█" * int(acc * 20)
            print(f"    {LABEL_NAMES[cls]:<15} : {acc:.0%}  {bar}")
        per_class_acc[LABEL_NAMES[cls]] = round(acc, 4)

    history["per_class_accuracy"] = per_class_acc
    history["val_predictions"]    = all_preds
    history["val_true_labels"]    = all_trues
    history["label_names"]        = [LABEL_NAMES[i] for i in range(NUM_CLASSES)]
    history["class_distribution"] = {
        LABEL_NAMES[i]: int((torch.tensor(np.load(Y_TRAIN_FILE)) == i).sum())
        for i in range(NUM_CLASSES)
    }

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Full history saved to: {HISTORY_FILE}")
    print("  Run step7_generate_graphs.py to create all graphs.\n")


if __name__ == "__main__":
    train()