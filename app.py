import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# make src importable
sys.path.append(str((Path(__file__).resolve().parents[1] / "src").resolve()))
from model_utils import load_model

st.set_page_config(page_title="Crop Recommendation", page_icon="🌾")

st.title("🌾 Crop Recommendation System")
st.caption("Enter soil & climate parameters to get a crop suggestion.")

# Load trained model
model = load_model()

with st.form("inputs"):
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Nitrogen (N)", 0.0, 200.0, 90.0, step=1.0)
        P = st.number_input("Phosphorus (P)", 0.0, 200.0, 42.0, step=1.0)
        K = st.number_input("Potassium (K)", 0.0, 200.0, 43.0, step=1.0)
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, step=0.1)
    with col2:
        temperature = st.number_input("Temperature (°C)", -10.0, 60.0, 21.0, step=0.1)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 82.0, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0, step=0.1)

    submitted = st.form_submit_button("Recommend Crop")

if submitted:
    row = pd.DataFrame([{
        "N": N, "P": P, "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    classes = model.classes_
    top3 = sorted(zip(classes, proba), key=lambda t: t[1], reverse=True)[:3]

    st.success(f"Recommended Crop: **{pred}**")
    st.subheader("Top 3 suggestions")
    for crop, p in top3:
        st.write(f"- {crop}: {p:.2%}")
