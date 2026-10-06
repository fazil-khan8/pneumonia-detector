import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import confusion_matrix

from data import get_loaders
from utils import get_device

device = get_device()
_, _, _, test_loader = get_loaders()

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device).eval()

probs, labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu()
        probs += p.tolist()
        labels += y.tolist()

for t in [0.5, 0.7, 0.8, 0.9, 0.95]:
    preds = [1 if p >= t else 0 for p in probs]
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    print(f"threshold {t}: missed sick = {fn}, false alarms = {fp}, accuracy = {(tp+tn)/len(labels):.3f}")