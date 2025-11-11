import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd
import altair as alt

# 🧭 PAGE SETTINGS
st.set_page_config(page_title="Coral Health Classifier", page_icon="🪸", layout="centered")

# 📂 MODEL PATH
MODEL_PATH = "CNN_model_MobileNet_CoralsClassification.h5"

# ⚙️ CHECK MODEL FILE
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found at `{MODEL_PATH}`. Please train and place it here.")
    st.stop()

# 🧩 LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
st.sidebar.title("⚙️ Settings")
st.sidebar.success("Model loaded successfully!")

# 🌗 THEME TOGGLE
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

theme_toggle = st.sidebar.toggle("🌓 Toggle Dark / Light Mode", value=(st.session_state.theme == "dark"))
st.session_state.theme = "dark" if theme_toggle else "light"

# 🎨 THEME VARIABLES
if st.session_state.theme == "dark":
    theme_vars = {
        "page_bg": "#081a2b",
        "card_bg": "#11263f",
        "expander_bg": "#1a324f",
        "text_color": "#e0f2f1",
        "accent_color": "#00d6b9",
        "border_color": "#00a88c",
        "chart_scheme": "tealblues"
    }
else:
    theme_vars = {
        "page_bg": "#f9fcfd",
        "card_bg": "#ffffff",
        "expander_bg": "#f1f5f9",
        "text_color": "#1a202c",
        "accent_color": "#00796b",
        "border_color": "#00796b",
        "chart_scheme": "greens"
    }

# 🧾 LOAD EXTERNAL CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Inject theme variables as CSS
st.markdown(f"""
<style>
:root {{
    --page-bg: {theme_vars["page_bg"]};
    --card-bg: {theme_vars["card_bg"]};
    --expander-bg: {theme_vars["expander_bg"]};
    --text-color: {theme_vars["text_color"]};
    --accent-color: {theme_vars["accent_color"]};
    --border-color: {theme_vars["border_color"]};
}}
</style>
""", unsafe_allow_html=True)

# 🪸 CLASS LABELS
class_names = ["Bleached", "Healthy"]

# ================================
# 🌊 APP HEADER
# ================================
st.markdown("<h1>🪸 Coral Health Classifier</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='text-align:center; font-size:1.15rem; color:{theme_vars['text_color']};'>"
    "Upload a coral reef image to determine if it’s <b>Healthy</b> or <b>Bleached</b> — "
    "and visualize the model’s confidence.</p>",
    unsafe_allow_html=True
)

# 📤 IMAGE UPLOAD
uploaded_file = st.file_uploader("🖼️ Upload a coral reef image", type=["jpg", "jpeg", "png"])

def load_and_prep_image(image, img_shape=224):
    img = image.resize((img_shape, img_shape))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# 🔍 PREDICTION
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    with st.spinner("🔬 Analyzing image..."):
        img = load_and_prep_image(image)
        pred = model.predict(img)
        bleached_prob = float(pred[0][0])
        healthy_prob = float(pred[0][1])
        confidence_gap = abs(bleached_prob - healthy_prob) * 100

    # 📊 Prediction Probabilities
    data = pd.DataFrame({
        'Class': class_names,
        'Probability': [bleached_prob, healthy_prob]
    })
    chart = (
        alt.Chart(data)
        .mark_bar(size=60)
        .encode(
            x=alt.X('Class', sort=None, title='Coral Health'),
            y=alt.Y('Probability', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('Class', scale=alt.Scale(scheme=theme_vars["chart_scheme"]))
        )
        .properties(height=300)
    )
    st.markdown("<h3>📊 Prediction Probabilities</h3>", unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True)

    # 🧠 INTERPRETATION
    st.markdown("<h3>🧠 Model Interpretation</h3>", unsafe_allow_html=True)

    if confidence_gap < 15:
        st.warning(f"⚠️ Low confidence — possible partial bleaching.\n\n"
                   f"Bleached: {bleached_prob*100:.2f}% | Healthy: {healthy_prob*100:.2f}%")
    elif bleached_prob > healthy_prob:
        st.error(f"⚪🪸 Prediction: Bleached Coral ({bleached_prob*100:.2f}% confidence)")
    else:
        st.success(f"🟢🪸 Prediction: Healthy Coral ({healthy_prob*100:.2f}% confidence)")

    # 🧩 MODEL DETAILS
    with st.expander("🧩 Model Details"):
        st.markdown(f"""
        <div style="color:{theme_vars["text_color"]}; font-size:1rem; line-height:1.6;">
        • Model Type: MobileNetV2 (CNN)<br>
        • Framework: TensorFlow / Keras<br>
        • Classes: Healthy 🟢🪸 & Bleached ⚪🪸<br>
        • Input Size: 224×224 RGB<br>
        • Output: Softmax probabilities<br>
        • Confidence Margin: {confidence_gap:.2f}%<br>
        • Purpose: Identify coral bleaching from underwater images.<br>
        • Disclaimer: Low confidence may indicate poor lighting, motion blur, or mixed coral conditions.
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(confidence_gap))

else:
    st.info("👆 Upload a coral reef image to begin classification.")

# 🪸 FOOTER
st.markdown("<hr>", unsafe_allow_html=True)
st.caption(f"<span style='color:{theme_vars['text_color']};'>🧠 Model: MobileNetV2 Coral Health Classifier • Built with TensorFlow & Streamlit</span>", unsafe_allow_html=True)
st.caption(f"<span style='color:{theme_vars['text_color']};'>🌊💙 Supporting coral reef conservation through AI.</span>", unsafe_allow_html=True)
