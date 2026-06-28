from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_pipeline.pkl"


def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}\n\nRun train.py first."
        )
    
    return joblib.load(MODEL_PATH)


def predict(patient_data):
    pipeline = load_pipeline()
    prediction = pipeline.predict(patient_data)
    return int(prediction[0])


def predict_probability(patient_data):
    pipeline = load_pipeline()
    probability = pipeline.predict_proba(patient_data)
    return float(probability[0][1])


def predict_with_probability(patient_data):
    pipeline = load_pipeline()
    prediction = int(pipeline.predict(patient_data)[0])
    probability = float(pipeline.predict_proba(patient_data)[0][1])

    return prediction, probability