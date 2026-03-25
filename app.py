"""
Heart Disease Detection - Streamlit Frontend
Input all 13 features and get prediction.
"""
import streamlit as st
import sys
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent to path for backend import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend.predict import predict

DATASET_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "training", "heart.csv"),
    os.path.join(os.path.dirname(__file__), "..", "training", "heart_disease_dataset.csv"),
]

COL_MAP = {
    "chest_pain_type": "cp",
    "resting_blood_pressure": "trestbps",
    "cholesterol": "chol",
    "fasting_blood_sugar": "fbs",
    "resting_ecg": "restecg",
    "max_heart_rate": "thalach",
    "exercise_induced_angina": "exang",
    "st_depression": "oldpeak",
    "st_slope": "slope",
    "num_major_vessels": "ca",
    "thalassemia": "thal",
    "heart_disease": "target",
}


@st.cache_data
def load_dataset():
    for path in DATASET_PATHS:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df.rename(columns=COL_MAP)
            if "target" not in df.columns and "num" in df.columns:
                df["target"] = (df["num"] > 0).astype(int)
            elif "target" not in df.columns:
                df["target"] = df.iloc[:, -1]
            return df
    return None


@st.cache_data
def load_best_model_name():
    metadata_path = os.path.join(os.path.dirname(__file__), "..", "model", "metadata.pkl")
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        return metadata.get("best_model_name")
    except Exception:
        return None


def get_feature_limits(df):
    defaults = {
        "age": (29, 77, 54),
        "trestbps": (94, 174, 130),
        "chol": (126, 336, 245),
        "thalach": (91, 202, 150),
        "oldpeak": (0.0, 4.9, 1.0),
    }
    if df is None:
        return defaults

    limits = {}
    for key in defaults:
        if key in df.columns:
            min_v = float(df[key].min())
            max_v = float(df[key].max())
            def_v = float(df[key].median())
            limits[key] = (min_v, max_v, def_v)
        else:
            limits[key] = defaults[key]
    return limits

st.set_page_config(
    page_title="Heart Disease Detection",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Detection")
st.markdown("Enter patient features to predict the risk of heart disease using ML classification.")
df = load_dataset()
limits = get_feature_limits(df)

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider(
            "Age (years)",
            int(limits["age"][0]),
            int(limits["age"][1]),
            int(limits["age"][2]),
        )
        sex = st.selectbox("Sex (0/1)", options=[0, 1])
        cp = st.selectbox(
            "Chest Pain Type (0-3)",
            options=[0, 1, 2, 3]
        )
        trestbps = st.slider(
            "Resting Blood Pressure (mm Hg)",
            int(limits["trestbps"][0]),
            int(limits["trestbps"][1]),
            int(limits["trestbps"][2]),
        )
        chol = st.slider(
            "Cholesterol (mg/dl)",
            int(limits["chol"][0]),
            int(limits["chol"][1]),
            int(limits["chol"][2]),
        )
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0/1)", options=[0, 1])
        restecg = st.selectbox(
            "Resting ECG (0-2)",
            options=[0, 1, 2]
        )

    with col2:
        thalach = st.slider(
            "Max Heart Rate Achieved",
            int(limits["thalach"][0]),
            int(limits["thalach"][1]),
            int(limits["thalach"][2]),
        )
        exang = st.selectbox("Exercise Induced Angina (0/1)", options=[0, 1])
        oldpeak = st.slider(
            "ST Depression (oldpeak)",
            float(limits["oldpeak"][0]),
            float(limits["oldpeak"][1]),
            float(limits["oldpeak"][2]),
            0.1,
        )
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment (0-2)",
            options=[0, 1, 2]
        )
        ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
        thal = st.selectbox(
            "Thalassemia (0-3)",
            options=[0, 1, 2, 3]
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        result = predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        pred = result['prediction']
        prob = result['probability']

        if pred == 1:
            st.error(f"🔴 **{result['label']}**")
            st.progress(prob)
            st.caption(f"Risk probability: {prob*100:.1f}%")
        else:
            st.success(f"🟢 **{result['label']}**")
            st.progress(1 - prob)
            st.caption(f"Probability of no disease: {(1-prob)*100:.1f}%")
    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
best_model_name = load_best_model_name()
if best_model_name:
    st.caption(f"Best trained model: {best_model_name}")
else:
    st.caption("Best model not available. Train the model to generate metadata.")

st.subheader("Visualizations")

if df is None:
    st.info("Dataset not found in training folder. Add `heart.csv` or `heart_disease_dataset.csv` to see charts.")
else:
    vis_col1, vis_col2 = st.columns(2)

    with vis_col1:
        st.markdown("**Target Distribution**")
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        df["target"].value_counts().sort_index().plot(
            kind="bar",
            ax=ax1,
            color=["#2ecc71", "#e74c3c"]
        )
        ax1.set_xlabel("Target (0 = No disease, 1 = Disease)")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis="x", rotation=0)
        st.pyplot(fig1, clear_figure=True)

    with vis_col2:
        st.markdown("**Age Distribution**")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        sns.histplot(df["age"], kde=True, bins=20, ax=ax2, color="#1f77b4")
        ax2.set_xlabel("Age")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2, clear_figure=True)

    st.markdown("**Correlation Heatmap**")
    numeric_df = df.select_dtypes(include=["number"])
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), cmap="RdYlBu_r", center=0, ax=ax3)
    st.pyplot(fig3, clear_figure=True)
