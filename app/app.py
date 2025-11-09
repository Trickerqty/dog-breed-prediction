import json
import io
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = APP_DIR / "assets"

# Try to auto-pick the newest run_* folder/checkpoint
def _find_latest_ckpt() -> Path | None:
    if not MODELS_DIR.exists():
        return None
    run_dirs = sorted([p for p in MODELS_DIR.glob("run_*") if p.is_dir()], reverse=True)
    for rd in run_dirs:
        ckpt = rd / "resnet50_dogbreed.pth"
        if ckpt.exists():
            return ckpt
    ckpt = MODELS_DIR / "resnet50_dogbreed.pth"
    return ckpt if ckpt.exists() else None

# ---------- Caching ----------
@st.cache_resource(show_spinner=False)
def load_model_and_classes() -> Tuple[nn.Module, List[str]]:
    ckpt_path = _find_latest_ckpt()
    if ckpt_path is None:
        raise FileNotFoundError(
            "No checkpoint found. Expected something like models/run_YYYYMMDD_HHMMSS/resnet50_dogbreed.pth "
            "or models/resnet50_dogbreed.pth"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=None)
    state = torch.load(ckpt_path, map_location=device)
    classes = state.get("classes", None)
    if classes is None:
        json_path = ASSETS_DIR / "class_names.json"
        if json_path.exists():
            classes = json.loads(json_path.read_text(encoding="utf-8"))
        else:
            raise RuntimeError(
                "Class names not found. Save them in the checkpoint under key 'classes' "
                "or provide app/assets/class_names.json"
            )
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(classes))
    model.load_state_dict(state["model_state"], strict=True)
    model.eval().to(device)

    return model, classes

@st.cache_resource(show_spinner=False)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

def predict_image(img_pil: Image.Image, model: nn.Module, classes: List[str], topk: int = 5):
    device = get_device()
    tfm = get_transform()
    with torch.no_grad():
        x = tfm(img_pil.convert("RGB")).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top_probs, top_idxs = torch.topk(probs, k=min(topk, len(classes)))
        top_probs = top_probs.cpu().numpy().tolist()
        top_idxs  = top_idxs.cpu().numpy().tolist()
        top_labels = [classes[i] for i in top_idxs]
    return list(zip(top_labels, top_probs))

# ---------- UI ----------
st.set_page_config(page_title="Dog Breed Prediction", page_icon="🐶", layout="centered")
st.title("🐶 Dog Breed Prediction (ResNet50 • PyTorch)")
st.caption("Upload a dog image to get the top-5 predicted breeds.")

# Load model once
try:
    model, class_names = load_model_and_classes()
    st.success(f"Model loaded. Classes: {len(class_names)}")
except Exception as e:
    st.error(str(e))
    st.stop()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
col1, col2 = st.columns([1,1])

with col1:
    if uploaded:
        try:
            img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
            st.image(img, caption="Uploaded image", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load image: {e}")
            st.stop()

with col2:
    if uploaded and st.button("Predict", type="primary", use_container_width=True):
        with st.spinner("Classifying..."):
            preds = predict_image(img, model, class_names, topk=5)
        st.subheader("Top-5 Predictions")
        for label, p in preds:
            st.write(f"• **{label}** — {p*100:.2f}%")
        st.bar_chart({label: p for label, p in preds})
