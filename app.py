import streamlit as st
import cv2
import numpy as np
import joblib
import plotly.graph_objects as go

from ensemble.deploy_ensemble import EnsembleModel
from ensemble.landmark_utils import compute_mediapipe_all_features
from ensemble.gradcam_resnet50 import load_resnet_gradcam, generate_gradcam

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Face Forgery Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM MODERN CSS THEME
# ============================================================
st.markdown("""
<style>

body {
    background: #f0f2f6;
}

/* Card style */
.stCard {
    padding: 1.2rem;
    background: white;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
    margin-bottom: 25px;
}

/* Section title */
.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 10px;
    color: #1f2937;
}

/* Probability label */
.pred-label {
    font-size: 22px;
    font-weight: 600;
    color: #374151;
}

/* Gauge center text */
.gauge-text {
    font-size: 32px !important;
    font-weight: 700 !important;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("<h1 style='text-align:center;margin-bottom:10px;'>Face Forgery Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:18px;color:#555;'>Deep Learning + Ensemble Models + Explainable AI</p>", unsafe_allow_html=True)

# ============================================================
# FILE UPLOAD CARD
# ============================================================
with st.container():
    st.markdown("<div class='stCard'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"])
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded:
    # ========================================================
    # READ IMAGE
    # ========================================================
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================
    with st.container():
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Step 1 — Extracting Landmark Features</div>", unsafe_allow_html=True)

        scaler = joblib.load("ensemble/scaler_mediapipe.pkl")
        mp_all = compute_mediapipe_all_features(img)
        mp_all = scaler.transform([mp_all])

        if mp_all is None:
            st.error("No face detected in the uploaded image.")
            st.stop()

        st.success("Landmark features extracted successfully.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # RUN ENSEMBLE PREDICTION
    # ========================================================
    model = EnsembleModel()
    result = model.predict(img, mp_all)

    prob_fake = result["final_prob"]
    label = "FAKE" if prob_fake > 0.5 else "REAL"
    color = "#ff3366" if label == "FAKE" else "#00b894"

    # ========================================================
    # GAUGE METER (Main Prediction)
    # ========================================================
    with st.container():
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Prediction Overview</div>", unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_fake * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"<b>{label}</b>", 'font': {'size': 36}},
            number={'suffix': "%", "font": {"size": 40}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': '#c8e6c9'},
                    {'range': [50, 100], 'color': '#ffcdd2'}
                ],
            }
        ))

        st.plotly_chart(fig_gauge, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # TABS FOR UI SECTIONS
    # ========================================================
    tab1, tab2, tab3 = st.tabs(["Image & Explainability", "Model Comparison", "Details"])

    # ========================================================
    # TAB 1 — IMAGE + GRAD-CAM
    # ========================================================
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='section-title'>Uploaded Image</div>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="content")

        with col2:
            st.markdown("<div class='section-title'>Grad-CAM Heatmap</div>", unsafe_allow_html=True)

            resnet_grad = load_resnet_gradcam("ensemble/resnet50_ensemble_mediapipe.pth")
            heatmap = generate_gradcam(resnet_grad, img)

            st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), width="content")

    # ========================================================
    # TAB 2 — MODEL COMPARISON
    # ========================================================
    with tab2:
        probs = {
            "SVM": result["svm"],
            "MLP": result["mlp"],
            "ResNet50": result["resnet"],
            "Ensemble": result["final_prob"]
        }

        st.markdown("<div class='section-title'>📊 Model Probability Comparison</div>", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(probs.keys()),
            y=list(probs.values()),
            marker=dict(color=["#3498db", "#e67e22", "#27ae60", "#e84393"]),
            text=[f"{p:.3f}" for p in probs.values()],
            textposition="outside"
        ))
        fig.update_layout(yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, width="stretch")

        if st.checkbox("Show Radar Plot"):
            categories = list(probs.keys())
            values = list(probs.values()) + [probs["SVM"]]

            fig2 = go.Figure(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                line_color="#e84393"
            ))
            fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
            st.plotly_chart(fig2, width="stretch")

    # ========================================================
    # TAB 3 — RAW DETAILS
    # ========================================================
    with tab3:
        st.subheader("📄 Raw Prediction Details")
        st.json(result)

    st.success("✔ Prediction Completed Successfully!")

