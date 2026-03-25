"""
Heart Disease Detection - Backend Prediction Logic
Loads the trained model and scaler, preprocesses input, returns prediction.
"""
import pickle
import os
import numpy as np

# Path to model files (relative to backend folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'heart_disease_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'model', 'scaler.pkl')
METADATA_PATH = os.path.join(BASE_DIR, '..', 'model', 'metadata.pkl')

_model = None
_scaler = None
_metadata = None


def _load_artifacts():
    """Lazy load model, scaler, and metadata."""
    global _model, _scaler, _metadata
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model not found. Please run training first and place "
                "heart_disease_model.pkl, scaler.pkl, metadata.pkl in the model folder."
            )
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            _scaler = pickle.load(f)
        with open(METADATA_PATH, 'rb') as f:
            _metadata = pickle.load(f)
    return _model, _scaler, _metadata


def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    """
    Predict heart disease from patient features.
    
    Returns:
        dict: {'prediction': 0 or 1, 'probability': float, 'label': str}
    """
    model, scaler, metadata = _load_artifacts()
    feature_cols = metadata['feature_columns']
    scale_cols = metadata.get('scale_columns', [])

    # Build feature vector in correct order
    feature_dict = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }

    # Handle missing columns (some datasets may have different columns)
    X = np.array([[feature_dict.get(c, 0) for c in feature_cols]], dtype=np.float64)

    # Apply scaling
    if scale_cols:
        scale_idx = [feature_cols.index(c) for c in scale_cols if c in feature_cols]
        X[:, scale_idx] = scaler.transform(X[:, scale_idx])

    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0, 1]) if hasattr(model, 'predict_proba') else float(pred)

    return {
        'prediction': pred,
        'probability': prob,
        'label': 'Heart Disease Detected' if pred == 1 else 'No Heart Disease'
    }
