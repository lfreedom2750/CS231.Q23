# app.py

import streamlit as st
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np

# =========================================================
# CONFIG
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Skin Disease Classification",
    layout="wide"
)

# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = [
    "Chickenpox",
    "Cowpox",
    "HFMD",
    "Healthy",
    "Measles",
    "Monkeypox"
]

NUM_CLASSES = len(CLASS_NAMES)

# =========================================================
# TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# CREATE MODELS
# =========================================================

def create_vgg16_model(num_classes):

    model = models.vgg16(weights=None)

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    return model


def create_resnet18_model(num_classes, use_dropout_head=False):

    model = models.resnet18(weights=None)

    in_features = model.fc.in_features

    if use_dropout_head:
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model(model_type, model_path):

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if model_type == "VGG16":
        model = create_vgg16_model(NUM_CLASSES)

    elif model_type == "ResNet18":
        use_dropout_head = "fc.1.weight" in state_dict
        model = create_resnet18_model(
            NUM_CLASSES,
            use_dropout_head=use_dropout_head
        )

    else:
        raise ValueError("Unsupported model")

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)
    model.eval()

    return model


# =========================================================
# PREDICT FUNCTION
# =========================================================

def predict_image(model, image):

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)

        probs = torch.softmax(outputs, dim=1)

        probs = probs.cpu().numpy()[0]

        pred_idx = np.argmax(probs)

        pred_class = CLASS_NAMES[pred_idx]

    return pred_class, probs


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Model Selection")

model_type = st.sidebar.selectbox(
    "Choose Model",
    ["VGG16", "ResNet18"]
)

# =========================================================
# MODEL PATHS
# =========================================================

MODEL_PATHS = {
    "VGG16": APP_DIR / "best_vgg16.pth",
    "ResNet18": APP_DIR / "best_resnet18.pth"
}

model_path = MODEL_PATHS[model_type]

# =========================================================
# LOAD MODEL
# =========================================================

model = load_model(model_type, model_path)

# =========================================================
# TITLE
# =========================================================

st.title("Skin Disease Classification")
st.markdown(f"### Current Model: `{model_type}`")

# =========================================================
# LAYOUT
# =========================================================

left_col, right_col = st.columns([1, 1])

# =========================================================
# UPLOAD IMAGE
# =========================================================

uploaded_file = left_col.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# DISPLAY RESULT
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # LEFT SIDE IMAGE
    left_col.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # PREDICTION
    pred_class, probs = predict_image(model, image)

    # CREATE DATAFRAME
    result_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability": probs
    })

    result_df["Probability"] = (
        result_df["Probability"] * 100
    ).round(2)

    result_df = result_df.sort_values(
        by="Probability",
        ascending=False
    )

    # RIGHT SIDE
    right_col.subheader("Prediction Result")

    right_col.success(
        f"Predicted Class: {pred_class}"
    )

    right_col.subheader("Soft Labels")

    # PROGRESS BAR
    for _, row in result_df.iterrows():

        class_name = row["Class"]
        prob = row["Probability"]

        right_col.markdown(
            f"<p style='font-size:13px'>{class_name}: {prob}%</p>",
            unsafe_allow_html=True
        )

        right_col.progress(float(prob / 100))

    # TABLE
    right_col.subheader("Probability Table")

    right_col.dataframe(
        result_df,
        use_container_width=True
    )

else:
    st.info("Please upload an image.")
