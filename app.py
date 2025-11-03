import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import time

# ----------------------------------
# 🎨 PAGE CONFIGURATION
# ----------------------------------
st.set_page_config(
    page_title="🌸 Period Health Analyzer",
    page_icon="🩸",
    layout="centered"
)

# ----------------------------------
# 🌈 CUSTOM STYLES
# ----------------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #ffe6eb, #fff0f5, #ffe4e1);
        font-family: 'Poppins', sans-serif;
    }
    h1 {
        color: #e91e63;
        text-align: center;
        font-size: 42px !important;
        font-weight: bold;
        text-shadow: 1px 1px 2px #f8bbd0;
    }
    .stButton>button {
        background-color: #e91e63;
        color: white;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6em 1.5em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #c2185b;
        transform: scale(1.05);
    }
    .stProgress > div > div > div {
        background-color: #f06292;
    }
    .card {
        background-color: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------
# 🧠 LOAD MODELS
# ----------------------------------
MAIN_MODEL_PATH = "period_blood_image_model.h5"

@st.cache_resource
def load_main_model():
    return tf.keras.models.load_model(MAIN_MODEL_PATH)

@st.cache_resource
def load_validator_model():
    return MobileNetV2(weights="imagenet")

try:
    model = load_main_model()
    validator_model = load_validator_model()
except Exception as e:
    st.error("❌ Model file not found! Please ensure 'period_blood_image_model.h5' is uploaded.")
    st.stop()

# ----------------------------------
# 🌸 APP HEADER
# ----------------------------------
st.markdown("<h1>🌸 Period Blood Image Analyzer</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:18px;'>Analyze menstrual blood images to check menstrual health condition and get awareness insights 🩸</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ----------------------------------
# 📷 IMAGE UPLOAD
# ----------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📸 Upload or Capture Image")

image_option = st.radio("Select Input Method:", ["📁 Upload from Device", "🎥 Use Camera"])
uploaded_img = None

if image_option == "📁 Upload from Device":
    uploaded_img = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
elif image_option == "🎥 Use Camera":
    uploaded_img = st.camera_input("Take a photo")

if uploaded_img:
    st.image(uploaded_img, caption="Preview of uploaded image", use_column_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------
# 🧍‍♀️ USER DETAILS FORM
# ----------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🩷 Personal & Menstrual Health Details")

with st.form("user_info_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.selectbox("Age Range", ["Under 18", "18-25", "26-35", "36-45", "Above 45"])
        weight = st.number_input("Weight (kg)", 30, 150, 55)
        height = st.number_input("Height (cm)", 120, 200, 160)
        flow = st.selectbox("Flow Amount", ["Light", "Moderate", "Heavy"])
    with col2:
        discharge = st.selectbox("Abnormal Discharge?", ["No", "Yes"])
        clotting = st.selectbox("Clotting During Periods?", ["No", "Yes"])
        pain = st.selectbox("Pain Management Method", ["No Pain", "Painkillers", "Hot Water Bag", "Rest", "Other"])
        product = st.selectbox("Menstrual Product", ["Pads", "Tampons", "Menstrual Cup", "Cloth", "Other"])

    freq_change = st.slider("Change Frequency (times/day)", 1, 10, 3)
    irritation = st.selectbox("Skin Irritation or Rashes?", ["No", "Yes"])
    awareness = st.selectbox("Aware of Menstrual Health?", ["Yes", "No"])
    anemia = st.selectbox("Diagnosed with Anemia?", ["No", "Yes"])

    submit_btn = st.form_submit_button("✨ Analyze Now")
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------
# 🧠 PREDICTION SECTION
# ----------------------------------
if submit_btn:
    if uploaded_img is None:
        st.warning("⚠️ Please upload or capture an image first.")
    else:
        with st.spinner("🔍 Validating and analyzing image..."):
            time.sleep(1)
            image = Image.open(uploaded_img).convert("RGB").resize((224, 224))

            # STEP 1 — Validate image with MobileNetV2
            img_check = np.expand_dims(np.array(image), axis=0)
            img_check = preprocess_input(img_check)
            preds_val = validator_model.predict(img_check)
            decoded_preds = decode_predictions(preds_val, top=5)[0]

            is_valid = any("blood" in name.lower() or "red" in name.lower() for (_, name, _) in decoded_preds)
            if not is_valid:
                st.warning("⚠️ This image doesn’t appear to be a menstrual blood image. Please upload a valid one.")
                st.stop()

            # STEP 2 — Predict using main model
            img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
            pred = model.predict(img_array)[0][0]

            confidence = abs(pred - 0.5) * 2  # measure confidence

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📊 Analysis Result")

            if confidence < 0.3:
                st.warning("⚠️ Model is uncertain about this image. Try a clearer image.")
            else:
                label = "🩷 Healthy" if pred < 0.5 else "💔 Unhealthy"
                st.success(f"**Result: {label}** (Confidence: {confidence*100:.1f}%)")
                st.image(image, caption="Analyzed Image", use_column_width=True)

                # Progress visualization
                st.progress(float(pred))
                time.sleep(0.5)

                st.balloons()  # 🎈 fun animation
            st.markdown("</div>", unsafe_allow_html=True)

            # STEP 3 — Show details
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📝 Summary of Your Details")
            st.write(f"**Age Range:** {age}")
            st.write(f"**Weight:** {weight} kg | **Height:** {height} cm")
            st.write(f"**Flow:** {flow} | **Discharge:** {discharge} | **Clotting:** {clotting}")
            st.write(f"**Pain Relief:** {pain} | **Product:** {product}")
            st.write(f"**Changes/day:** {freq_change} | **Irritation:** {irritation}")
            st.write(f"**Awareness:** {awareness} | **Anemia:** {anemia}")

            st.info("💡 *Tip:* Maintain hydration, balanced diet, and regular exercise to improve menstrual health.")
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>🩸 This tool is for educational use only. Consult a doctor for medical advice.</p>",
    unsafe_allow_html=True
)
