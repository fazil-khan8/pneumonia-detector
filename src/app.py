import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random
from glob import glob

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
from matplotlib import colormaps
from torchvision import models
from PIL import Image

from data import eval_tf, DATA_DIR
from utils import get_device

CLASSES = ["NORMAL", "PNEUMONIA"]

# Your real test-set results (624 images) from evaluate.py and threshold.py
CONF = [[155, 79], [2, 388]]
THRESH = pd.DataFrame({
    "threshold": [0.5, 0.7, 0.8, 0.9, 0.95],
    "missed_sick": [2, 4, 5, 7, 11],
    "false_alarms": [79, 67, 62, 54, 45],
    "accuracy": [87.0, 88.6, 89.3, 90.2, 91.0],
})

st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.result-card {padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: .8rem;}
.result-pos {background: rgba(239,68,68,.15); border: 1px solid rgba(239,68,68,.55);}
.result-neg {background: rgba(34,197,94,.15); border: 1px solid rgba(34,197,94,.55);}
.result-card h2 {margin: 0; font-size: 1.9rem;}
.result-card p {margin: .3rem 0 0; opacity: .85;}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- model
@st.cache_resource
def load_model():
    device = get_device()
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    return model.to(device).eval(), device


model, device = load_model()


def predict(img):
    x = eval_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu()
    return x, probs


def gradcam(x, class_idx):
    """Grad-CAM on the last conv block of ResNet18."""
    store = {}

    def fwd_hook(_, __, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = model.layer4.register_forward_hook(fwd_hook)
    model.zero_grad()
    out = model(x)
    out[0, class_idx].backward()
    handle.remove()

    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy()


def overlay(img, cam, alpha):
    size = (448, 448)
    base = np.asarray(img.convert("L").resize(size).convert("RGB"), dtype=np.float32) / 255
    cam_big = np.asarray(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(size, Image.BILINEAR),
        dtype=np.float32,
    ) / 255
    heat = colormaps["jet"](cam_big)[..., :3]
    w = (alpha * cam_big)[..., None]
    return ((base * (1 - w) + heat * w) * 255).astype(np.uint8)


def looks_like_xray(img):
    """Rough check: X-rays are almost grayscale, most photos are colorful."""
    arr = np.asarray(img.resize((128, 128)), dtype=np.float32)
    color_diff = np.abs(arr[..., 0] - arr[..., 1]).mean() + np.abs(arr[..., 1] - arr[..., 2]).mean()
    return color_diff < 8


# ---------------------------------------------------------------- charts
def gauge(p, thr):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Pneumonia probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#ef4444" if p >= thr else "#22c55e"},
            "threshold": {"line": {"color": "white", "width": 4},
                          "thickness": 0.85, "value": thr * 100},
        },
    ))
    fig.update_layout(height=250, margin=dict(l=25, r=25, t=60, b=10))
    return fig


def confusion_fig():
    fig = go.Figure(go.Heatmap(
        z=CONF,
        x=["Predicted NORMAL", "Predicted PNEUMONIA"],
        y=["True NORMAL", "True PNEUMONIA"],
        colorscale="Teal", showscale=False,
        text=CONF, texttemplate="%{text}", textfont={"size": 30},
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def threshold_fig(current):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=THRESH.threshold, y=THRESH.false_alarms, mode="lines+markers",
                             name="False alarms (healthy flagged)", line=dict(color="#f59e0b", width=3)))
    fig.add_trace(go.Scatter(x=THRESH.threshold, y=THRESH.missed_sick, mode="lines+markers",
                             name="Missed sick patients", line=dict(color="#ef4444", width=3)))
    fig.add_vline(x=current, line_dash="dash", line_color="white",
                  annotation_text="your threshold", annotation_position="top")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Decision threshold", yaxis_title="Number of images (of 624)",
                      legend=dict(orientation="h", y=-0.25))
    return fig


@st.cache_data
def dataset_counts():
    rows = []
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            d = f"{DATA_DIR}/{split}/{cls}"
            n = len([f for f in os.listdir(d) if not f.startswith(".")]) if os.path.isdir(d) else 0
            rows.append({"split": split, "class": cls, "images": n})
    return pd.DataFrame(rows)


def dataset_fig(df):
    fig = go.Figure()
    colors = {"NORMAL": "#22c55e", "PNEUMONIA": "#ef4444"}
    for cls in CLASSES:
        sub = df[df["class"] == cls]
        fig.add_trace(go.Bar(x=sub.split, y=sub.images, name=cls,
                             marker_color=colors[cls], text=sub.images, textposition="outside"))
    fig.update_layout(barmode="group", height=360, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Images", legend=dict(orientation="h", y=-0.2))
    return fig


# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Controls")
threshold = st.sidebar.slider(
    "Pneumonia threshold", 0.30, 0.95, 0.50, 0.05,
    help="Higher = fewer false alarms but more missed pneumonia cases.",
)
show_cam = st.sidebar.toggle("Show Grad-CAM heatmap", value=True)
alpha = st.sidebar.slider("Heatmap strength", 0.2, 1.0, 0.7, 0.1, disabled=not show_cam)
st.sidebar.divider()
st.sidebar.caption(f"Model: ResNet18 (transfer learning)\n\nRunning on: **{device}**")
st.sidebar.warning("Educational project. NOT a medical device.")

# ---------------------------------------------------------------- header
st.title("🫁 Chest X-Ray Pneumonia Detector")
st.caption("Upload a chest X-ray, get a prediction, and see where the model was looking.")




tab_analyze, tab_insights, tab_about = st.tabs(["🔍 Analyze", "📊 Model Insights", "ℹ️ About"])

# ---------------------------------------------------------------- analyze
with tab_analyze:
    source = st.radio("Image source", ["Upload X-ray", "Random test image"], horizontal=True)
    img, truth = None, None

    if source == "Upload X-ray":
        f = st.file_uploader("Upload a chest X-ray", type=["jpg", "jpeg", "png"])
        if f:
            img = Image.open(f).convert("RGB")
    else:
        pool = glob(f"{DATA_DIR}/test/*/*.jp*g")
        if not pool:
            st.error("No test images found. Run the app from the project root.")
        else:
            if "sample" not in st.session_state:
                st.session_state.sample = random.choice(pool)
            if st.button("🎲 Pick another random image"):
                st.session_state.sample = random.choice(pool)
            path = st.session_state.sample
            truth = os.path.basename(os.path.dirname(path))
            img = Image.open(path).convert("RGB")

    if img is not None and not looks_like_xray(img):
        st.warning("This image has color, so it probably is not a chest X-ray. "
                   "The model was only trained on X-rays, so its answer would be meaningless.")
    if not st.checkbox("Analyze anyway"):
            img = None


    
    if img is None:
        st.info("👆 Upload an image or pick a random test image to start.")
    else:
        x, probs = predict(img)
        p_pneu = probs[1].item()
        label = "PNEUMONIA" if p_pneu >= threshold else "NORMAL"
        label_idx = CLASSES.index(label)

        left, right = st.columns([3, 2], gap="large")

        with left:
            c1, c2 = st.columns(2)
            c1.image(img, caption="Original", use_container_width=True)
            if show_cam:
                cam = gradcam(x, label_idx)
                c2.image(overlay(img, cam, alpha),
                         caption=f"Grad-CAM: what pushed the model toward {label}",
                         use_container_width=True)
            else:
                c2.info("Heatmap is turned off in the sidebar.")
            if show_cam:
                st.caption("Warm colors (red/yellow) = regions that influenced the decision most. "
                           "If they sit on borders or text markers instead of the lungs, don't trust the prediction.")

        with right:
            css = "result-pos" if label == "PNEUMONIA" else "result-neg"
            icon = "⚠️" if label == "PNEUMONIA" else "✅"
            st.markdown(
                f'<div class="result-card {css}"><h2>{icon} {label}</h2>'
                f'<p>Pneumonia probability {p_pneu * 100:.1f}% vs threshold {threshold * 100:.0f}%</p></div>',
                unsafe_allow_html=True,
            )
            if truth:
                ok = truth == label
                (st.success if ok else st.error)(
                    f"Ground truth: **{truth}**. The model is {'correct ✔' if ok else 'wrong ✘'}."
                )
            st.plotly_chart(gauge(p_pneu, threshold), use_container_width=True)

            st.dataframe(
                pd.DataFrame({"Class": CLASSES, "Probability": [f"{p.item() * 100:.2f}%" for p in probs]}),
                hide_index=True, use_container_width=True,
            )
            if p_pneu >= threshold and p_pneu < 0.7:
                st.warning("Borderline case: the model is not very sure about this one.")
            st.caption("Neural networks are often overconfident. Treat this as a rough signal.")

# ---------------------------------------------------------------- insights
with tab_insights:
    st.subheader("How well does the model work?")
    st.caption("Measured on 624 unseen test images (threshold 0.5).")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "87.0%")
    m2.metric("Pneumonia recall", "99.5%", help="388 of 390 sick patients were caught.")
    m3.metric("Normal recall", "66.2%", help="155 of 234 healthy X-rays were called normal.")
    m4.metric("Normal precision", "98.7%", help="When the model says NORMAL, it is right 98.7% of the time.")

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### Confusion matrix")
        st.plotly_chart(confusion_fig(), use_container_width=True)
        st.caption("Bottom-left (2) = sick people missed. Top-right (79) = healthy people wrongly flagged.")
    with b:
        st.markdown("#### Threshold trade-off")
        st.plotly_chart(threshold_fig(threshold), use_container_width=True)
        st.caption("Raising the threshold removes false alarms but misses more sick patients.")

    st.markdown("#### Threshold table")
    st.dataframe(
        THRESH.rename(columns={"threshold": "Threshold", "missed_sick": "Missed sick",
                               "false_alarms": "False alarms", "accuracy": "Accuracy %"}),
        hide_index=True, use_container_width=True,
    )

    st.markdown("#### Dataset balance")
    df = dataset_counts()
    c, d = st.columns([3, 2], gap="large")
    with c:
        st.plotly_chart(dataset_fig(df), use_container_width=True)
    with d:
        tr = df[df.split == "train"].set_index("class")["images"]
        if tr.get("NORMAL", 0) > 0:
            ratio = tr["PNEUMONIA"] / tr["NORMAL"]
            st.metric("Train imbalance", f"{ratio:.1f} : 1", help="Pneumonia images per normal image")
        st.markdown(
            "**Key takeaways**\n"
            "- The model is **cautious**: it almost never misses pneumonia.\n"
            "- The price is **false alarms** on healthy lungs.\n"
            "- Training data has about 3x more pneumonia than normal, so the model leans toward pneumonia. "
            "Class weights reduce this but don't remove it.\n"
            "- The test images come from a different distribution than the training images, "
            "which also hurts NORMAL recall."
        )

# ---------------------------------------------------------------- about
with tab_about:
    st.subheader("How it works")
    st.markdown(
        "1. The X-ray is resized to 224x224 and normalized.\n"
        "2. A **ResNet18** pretrained on ImageNet, fine-tuned on the Kaggle pneumonia dataset, outputs two scores.\n"
        "3. A softmax turns the scores into probabilities.\n"
        "4. The **threshold** decides when to call PNEUMONIA.\n"
        "5. **Grad-CAM** uses the gradients of the last convolutional block to show which image regions "
        "mattered for the decision."
    )
    st.subheader("Limitations")
    st.markdown(
        "- Educational project, not for diagnosis.\n"
        "- Trained on one dataset from one source. It may fail on X-rays from other hospitals or scanners.\n"
        "- It cannot tell bacterial from viral pneumonia, and cannot detect other diseases.\n"
        "- Threshold choices were examined on the test set, so reported numbers are slightly optimistic.\n"
        "- Heatmaps show correlation with the model's decision, not medical proof."
    )




















