import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix

from data import get_loaders
from utils import get_device

device = get_device()
print("Using device:", device)

_, _, _, test_loader = get_loaders()

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device).eval()

preds, labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        out = model(x.to(device)).argmax(1).cpu()
        preds += out.tolist()
        labels += y.tolist()

print(classification_report(labels, preds, target_names=["NORMAL", "PNEUMONIA"], digits=3))
print("Confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(labels, preds))