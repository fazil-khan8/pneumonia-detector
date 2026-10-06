import os, sys
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

from data import eval_tf
from utils import get_device

device = get_device()
classes = ["NORMAL", "PNEUMONIA"]

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device).eval()

img = Image.open(sys.argv[1]).convert("RGB")
x = eval_tf(img).unsqueeze(0).to(device)

with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]

print(f"Prediction: {classes[probs.argmax().item()]} ({probs.max().item() * 100:.1f}%)")