import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import gdown
import os
from model import XuNetSteganalysis

MODELS = {
    "Best 1v3 - HILL trained": {
        "id": "1KWCq9mjsrpqbiPLYMf-tc3J-a_GiwG9P",
        "desc": "Trained on HILL steganography. Specialised for detecting one algorithm.",
        "algorithms": ["HILL"],
    },
    "Best 2v2 - HILL+WOW trained": {
        "id": "13XHfWIsFpDPPOGzJRFIEUQFG9k6kx5k9",
        "desc": "Trained on HILL and WOW. Balanced multi-algorithm detection.",
        "algorithms": ["HILL", "WOW"],
    },
    "Best 3v1 - S-UNIWARD+HILL+WOW trained": {
        "id": "1BfuemEU6FQV151nFVbg_bjSuYjr4YW3B",
        "desc": "Trained on S-UNIWARD, HILL, and WOW. Broadest coverage.",
        "algorithms": ["S-UNIWARD", "HILL", "WOW"],
    },
}

@st.cache_resource
def load_model(model_name):
    file_id = MODELS[model_name]
    output_path = f"model_{list(MODELS.keys()).index(model_name)}.pth"

    if not os.path.exists(output_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

    model = XuNetSteganalysis()
    model.load_state_dict(torch.load(output_path, map_location=torch.device('cpu')))
    model.eval()

    return model

def preprocess(image):
    image = image.convert('L')                                      # Convert to grayscale
    image = image.resize((512, 512))                                # Resize to 512x512
    tensor = torch.tensor(np.array(image), dtype=torch.float32)     # Convert to tensor
    tensor = tensor / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)                       # add batch and channel dimensions
    return tensor

def predict(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1)
        cover_prob = probabilities[0][0].item()
        stego_prob = probabilities[0][1].item()
    return cover_prob, stego_prob, elapsed

with st.sidebar:
    st.header("Model")
    selected_model = st.selectbox("Select a model:", list(MODELS.keys()), label_visibility="collapsed")
    info = MODELS[selected_model]
    st.caption(info["desc"])
    st.write("**Detects:**", " · ".join(info["algorithms"]))

    st.divider()
    st.header("About")
    st.write(
        "This demo uses the **XuNet** CNN architecture (Xu et al., 2016) "
        "to classify whether an image contains hidden data embedded via spatial-domain steganography."
    )
    st.write("Upload a grayscale or colour JPEG/PNG/PGM and the model will output a cover vs. stego probability.")

st.title("CNN-Based Steganalysis Demo")
st.caption("Detect hidden data in images using deep learning")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "pgm"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    original_size = uploaded_file.size

    col_img, col_results = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)
        with st.expander("Image details"):
            w, h = image.size
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Dimensions:** {w} × {h} px")
            st.write(f"**Colour mode:** {image.mode}")
            st.write(f"**File size:** {original_size / 1024:.1f} KB")

    with col_results:
        st.subheader("Analysis")

        with st.spinner("Running inference…"):
            model = load_model(selected_model)
            tensor = preprocess(image)
            cover_prob, stego_prob, elapsed_s = predict(model, tensor)

        if stego_prob > 0.5:
            st.error(f"Stego image detected — {stego_prob:.1%} confidence", icon="🚨")
        else:
            st.success(f"Cover image (clean) — {cover_prob:.1%} confidence", icon="✅")

        st.divider()

        m1, m2 = st.columns(2)
        m1.metric("Cover probability", f"{cover_prob:.1%}")
        m2.metric("Stego probability", f"{stego_prob:.1%}")

        st.progress(cover_prob, text="Cover")
        st.progress(stego_prob, text="Stego")

        st.caption(f"Inference time: {elapsed_s * 1000:.0f} ms · Model: {selected_model}")
