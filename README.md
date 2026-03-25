# Heart Disease Detection

A machine learning classification project that predicts heart disease from patient features using Decision Tree, Random Forest, Logistic Regression, and SVM.

## Project Structure

```
p2/
├── frontend/          # Streamlit app
│   └── app.py
├── backend/           # Prediction logic
│   ├── __init__.py
│   └── predict.py
├── model/             # Trained model & scaler (.pkl files)
│   ├── heart_disease_model.pkl
│   ├── scaler.pkl
│   └── metadata.pkl
├── training/          # Training scripts & dataset
│   ├── train.py
│   ├── heart_disease_training.ipynb
│   └── heart.csv      # Place your dataset here
├── requirements.txt
└── README.md
```

## Setup Steps

### 1. Place Dataset
Put `heart.csv` in the `training/` folder. Columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target

### 2. Train the Model (Google Colab or Local)
- **Option A:** Upload `training/heart_disease_training.ipynb` to Colab, upload `heart.csv`, run all cells, download the 3 `.pkl` files
- **Option B:** Run locally: `python training/train.py`

### 3. Place Model Files
Put `heart_disease_model.pkl`, `scaler.pkl`, and `metadata.pkl` in the `model/` folder.

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the App
```bash
streamlit run frontend/app.py
```
Open http://localhost:8501

## Deployment (Render)

1. Push to GitHub
2. Create a new Web Service on Render
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0`
5. Ensure model files are in the repo (or add them via secrets/build)
