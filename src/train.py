import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"   # must be before import torch

import torch
import torch.nn as nn
from torchvision import models
from collections import Counter
from tqdm import tqdm

from data import get_loaders
from utils import get_device

device = get_device()
print("Using device:", device)

train_ds, train_loader, val_loader, _ = get_loaders(batch_size=32)

# Pretrained ResNet18, replace the last layer with 2 classes
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Dataset is imbalanced (more pneumonia), so weight the classes
counts = Counter(train_ds.targets)
weights = torch.tensor([1 / counts[0], 1 / counts[1]], dtype=torch.float)
weights = (weights / weights.sum() * 2).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 5
best_acc = 0.0
os.makedirs("models", exist_ok=True)

for epoch in range(EPOCHS):
    # ---- train ----
    model.train()
    running_loss = 0.0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Train loss: {running_loss / len(train_loader):.4f}")

    # ---- validate ----
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"Val accuracy: {acc:.3f}")

    if acc >= best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "models/best_model.pth")
        print("Saved best model")

print("Done. Best val accuracy:", best_acc)