import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import gdown
import os
from model import XuNetSteganalysis

MODELS = {
    "Best 1v3 - HILL trained": "1KWCq9mjsrpqbiPLYMf-tc3J-a_GiwG9P",
    "Best 2v2 - HILL+WOW trained": "13XHfWIsFpDPPOGzJRFIEUQFG9k6kx5k9",
    "Best 3v1 - S-UNIWARD+HILL+WOW trained": "1BfuemEU6FQV151nFVbg_bjSuYjr4YW3B"
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
    tensor = tensor.unsqueeze(0).unsqueeze(0)                       # add batch and channel dimensions
    return tensor

def predict(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1)
        cover_prob = probabilities[0][0].item()
        stego_prob = probabilities[0][1].item()
    return cover_prob, stego_prob

st.title("CNN-Based Steganalysis Demo")

selected_model = st.selectbox("Select a model:", list(MODELS.keys()))

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "pgm"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model(selected_model)
    tensor = preprocess(image)
    cover_prob, stego_prob = predict(model, tensor)

    st.subheader("Prediction")
    if stego_prob > 0.5:
        st.error(f"Stego Image Detected! - Confidence: {stego_prob:.1%}")
    else:
        st.success(f"Cover Image - Confidence: {cover_prob:.1%}")

    st.subheader("Confidence")
    st.write("Cover Probability")
    st.progress(cover_prob)
    st.write("Stego Probability")
    st.progress(stego_prob)
