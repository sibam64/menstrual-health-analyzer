# app.py
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# MobileNet validator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Optional download helper
try:
    import gdown
except Exception:
    gdown = None

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="🌸 Period Health Analyzer", page_icon="🩸", layout="centered")

# ------------------------------
# STYLE (keeps same look)
# ------------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #ffe6eb, #fff0f5, #ffe4e1);
        font-family: 'Poppins', sans-serif;
    }
    h1 { color: #e91e63; text-align: center; font-size: 42px !important; font-weight: bold; text-shadow: 1px 1px 2px #f8bbd0; }
    .stButton>button { background-color: #e91e63; color: white; border-radius: 10px; font-weight: 600; padding: 0.6em 1.5em; transition: all 0.3s ease; }
    .stButton>button:hover { background-color: #c2185b; transform: scale(1.03); }
    .stProgress > div > div > div { background-color: #f06292; }
    .card { background-color: white; border-radius: 15px; padding: 18px; box-shadow: 0px 4px 15px rgba(0,0,0,0.08); margin-bottom: 16px; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# MODEL PATHS & DRIVE FALLBACK
# ------------------------------
MAIN_MODEL_FILENAME = "period_blood_image_model.h5"   # expected local filename
MAIN_MODEL_PATH = os.path.join(os.getcwd(), MAIN_MODEL_FILENAME)

# Optional: If your model is too large to commit to GitHub,
# put the Google Drive file id into Streamlit secrets as:
# [secrets]
# DRIVE_MODEL_FILE_ID = "1abcD...xyz"
DRIVE_FILE_ID = st.secrets.get("DRIVE_MODEL_FILE_ID", None)

# ------------------------------
# LOAD MODELS (with caching)
# ------------------------------
@st.cache_resource
def load_validator_model():
    """MobileNetV2 pretrained on ImageNet for quick validation"""
    return MobileNetV2(weights="imagenet")

@st.cache_resource
def load_main_model(path):
    """Load the main .h5 model from disk"""
    return tf.keras.models.load_model(path)

def download_model_from_drive(file_id: str, out_path: str) -> bool:
    """Download using gdown. Returns True on success."""
    if gdown is None:
        st.error("gdown is not installed on the server. Add 'gdown' to requirements.txt or upload model to repo.")
        return False
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        st.info("Downloading model from Google Drive (this may take a minute)...")
        gdown.download(url, out_path, quiet=False)
        return os.path.exists(out_path)
    except Exception as e:
        st.error(f"Failed to download model from Drive: {e}")
        return False

# Ensure model exists locally or attempt download
if not os.path.exists(MAIN_MODEL_PATH):
    if DRIVE_FILE_ID:
        success = download_model_from_drive(DRIVE_FILE_ID, MAIN_MODEL_PATH)
        if not success:
            st.error("Model not found locally and Drive download failed. Please upload the model file or set correct DRIVE_MODEL_FILE_ID in Streamlit secrets.")
            st.stop()
    else:
        st.warning("Model file not found. You can either upload 'period_blood_image_model.h5' to the app folder or set DRIVE_MODEL_FILE_ID in Streamlit secrets.")
        st.stop()

# Try loading models
try:
    model = load_main_model(MAIN_MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load main model ({MAIN_MODEL_FILENAME}). Check file and TensorFlow compatibility. Details: {e}")
    st.stop()

try:
    validator_model = load_validator_model()
except Exception as e:
    st.error(f"Failed to load image validator model. Details: {e}")
    st.stop()

# ------------------------------
# APP UI
# ------------------------------
st.markdown("<h1>🌸 Period Blood Image Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:16px;'>Upload or capture an image and answer a few quick questions. For educational use only.</p>", unsafe_allow_html=True)
st.markdown("---")

# Input
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📸 Upload or Capture Image")
input_choice = st.radio("Input method:", ["Upload from device", "Use camera"])

uploaded = None
if input_choice == "Upload from device":
    uploaded = st.file_uploader("Choose an image (jpg, png)", type=["jpg","jpeg","png"])
else:
    # camera_input will request permission only when used
    open_cam = st.button("Open Camera")
    if open_cam:
        uploaded = st.camera_input("Take a photo")

if uploaded:
    st.image(uploaded, caption="Preview", use_column_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Form
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🩷 Personal & Menstrual Health Details")
with st.form("info_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.selectbox("Age Range", ["Under 18","18-25","26-35","36-45","Above 45"])
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=60)
        height = st.number_input("Height (cm)", min_value=120, max_value=220, value=160)
        flow = st.selectbox("Flow Amount", ["Light","Moderate","Heavy"])
    with c2:
        discharge = st.selectbox("Abnormal Discharge?", ["No","Yes"])
        clotting = st.selectbox("Clotting during period?", ["No","Yes"])
        pain = st.selectbox("Pain management", ["No Pain","Painkillers","Hot Water Bag","Rest","Other"])
        product = st.selectbox("Product used", ["Pads","Tampons","Menstrual Cup","Cloth","Other"])
    freq_change = st.slider("Change frequency (times/day)", 1, 12, 3)
    irritation = st.selectbox("Skin irritation/rashes?", ["No","Yes"])
    awareness = st.selectbox("Aware of abnormal signs?", ["Yes","No"])
    anemia = st.selectbox("Diagnosed with anemia?", ["No","Yes"])
    submit = st.form_submit_button("Analyze Now")
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# ANALYSIS
# ------------------------------
if submit:
    if uploaded is None:
        st.warning("Please upload or capture an image first.")
    else:
        with st.spinner("Validating image and running model..."):
            time.sleep(0.8)
            # Load image in PIL and resize
            img = Image.open(uploaded).convert("RGB").resize((224,224))
            # Step 1: check with MobileNet validator
            img_arr = np.expand_dims(np.array(img), axis=0)
            img_pre = preprocess_input(img_arr.copy())
            try:
                val_preds = validator_model.predict(img_pre)
                decoded = decode_predictions(val_preds, top=5)[0]
            except Exception as e:
                st.error(f"Validator model prediction failed: {e}")
                st.stop()

            # heuristic: check for words "blood" or "red" in top predictions
            is_blood_like = any(("blood" in name.lower() or "red" in name.lower()) for (_, name, _) in decoded)
            if not is_blood_like:
                # fallback: if user insists, allow override button
                st.warning("This image does not look like a menstrual blood image. If you believe this is a valid image, click 'Force Analyze'.")
                if st.button("Force Analyze (override)"):
                    is_blood_like = True
                else:
                    st.stop()

            # Step 2: main model prediction (normalized)
            x = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)
            try:
                pred = model.predict(x)[0][0]
            except Exception as e:
                st.error(f"Main model prediction failed: {e}")
                st.stop()

            confidence = abs(pred - 0.5) * 2  # 0..1
            if confidence < 0.3:
                st.warning("Model uncertain (low confidence). Try a clearer sample.")
            else:
                label = "🩷 Healthy" if pred < 0.5 else "💔 Unhealthy"
                st.success(f"Result: {label} (Confidence: {confidence*100:.1f}%)")
                st.image(img, caption="Analyzed image", use_column_width=True)
                st.progress(float(pred))

            # summary
            st.markdown("---")
            st.subheader("Summary")
            st.write(f"**Age:** {age} — **Weight:** {weight} kg — **Height:** {height} cm")
            st.write(f"**Flow:** {flow} — **Discharge:** {discharge} — **Clotting:** {clotting}")
            st.write(f"**Pain management:** {pain} — **Product:** {product} — **Change/day:** {freq_change}")
            st.write(f"**Irritation:** {irritation} — **Awareness:** {awareness} — **Anemia:** {anemia}")
            st.markdown("---")
            st.info("This tool is for educational purposes only. Consult a healthcare professional for medical diagnosis.")
