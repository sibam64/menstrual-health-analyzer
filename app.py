import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="🌸 Period Health Analyzer",
    layout="centered",
    page_icon="🩸"
)

# ------------------------------
# LOAD MODEL
# ------------------------------
MODEL_PATH = "period_blood_image_model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("❌ Model file not found! Please ensure the model is uploaded as 'period_blood_image_model.h5'.")
    st.stop()

# ------------------------------
# APP HEADER
# ------------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #e91e63;'>🌸 Period Blood Image Analyzer</h1>
    <p style='text-align: center;'>Analyze menstrual blood image and health indicators to help assess menstrual wellness.</p>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# IMAGE INPUT SECTION
# ------------------------------
st.subheader("📷 Upload or Capture Image")

image_option = st.radio("Choose Image Input Method:", ["Upload from Device", "Use Camera"])

uploaded_img = None
if image_option == "Upload from Device":
    uploaded_img = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
elif image_option == "Use Camera":
    uploaded_img = st.camera_input("Take a photo")

# ------------------------------
# USER DETAILS FORM
# ------------------------------
st.subheader("🧍‍♀️ Personal & Menstrual Health Information")

with st.form("user_info_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.selectbox("Age Range", ["Under 18", "18-25", "26-35", "36-45", "Above 45"])
        weight = st.number_input("Weight (kg)", min_value=30, max_value=150, step=1)
        height = st.number_input("Height (cm)", min_value=120, max_value=200, step=1)
        flow = st.selectbox("Flow Amount", ["Light", "Moderate", "Heavy"])
    with col2:
        discharge = st.selectbox("Presence of Abnormal Discharge?", ["No", "Yes"])
        clotting = st.selectbox("Clotting During Periods?", ["No", "Yes"])
        pain = st.selectbox("How do you usually manage menstrual pain?", 
                            ["No Pain", "Painkillers", "Hot Water Bag", "Rest", "Other"])
        product = st.selectbox("Menstrual Product Used", 
                               ["Pads", "Tampons", "Menstrual Cup", "Cloth", "Other"])
    
    freq_change = st.slider("How often do you change your product per day?", 1, 10, 3)
    irritation = st.selectbox("Any skin irritation or rashes?", ["No", "Yes"])
    awareness = st.selectbox("Are you aware of abnormal menstrual health signs?", ["Yes", "No"])
    anemia = st.selectbox("Diagnosed with Anemia?", ["No", "Yes"])
    
    submit_btn = st.form_submit_button("✅ Submit Details & Analyze")

# ------------------------------
# PREDICTION SECTION
# ------------------------------
if submit_btn:
    if uploaded_img is None:
        st.warning("⚠️ Please upload or capture an image first.")
    else:
        st.info("🧠 Analyzing your image... Please wait.")
        image = Image.open(uploaded_img).convert("RGB").resize((224, 224))
        img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
        pred = model.predict(img_array)[0][0]
        label = "🩷 Healthy" if pred < 0.5 else "💔 Unhealthy"

        st.success(f"**Result: {label}**")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.progress(float(pred))

        # Display summary
        st.subheader("📋 Summary of your details")
        st.write(f"**Age Range:** {age}")
        st.write(f"**Weight:** {weight} kg, **Height:** {height} cm")
        st.write(f"**Flow:** {flow}, **Discharge:** {discharge}, **Clotting:** {clotting}")
        st.write(f"**Pain Management:** {pain}")
        st.write(f"**Product Used:** {product}, **Changed per day:** {freq_change}")
        st.write(f"**Irritation:** {irritation}, **Awareness:** {awareness}, **Anemia:** {anemia}")

        st.markdown("---")
        st.markdown(
            "<p style='text-align:center; color:gray;'>🩸 This tool is for educational use only. Consult a medical professional for diagnosis.</p>",
            unsafe_allow_html=True
        )
