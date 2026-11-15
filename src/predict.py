from pathlib import Path
from typing import Dict, List, Union
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

TARGET_INV_MAPPING = {0: 'Low', 1: 'Medium', 2: 'High'}


def load_artifacts():
    """Load preprocessor and best model."""
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}")
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {BEST_MODEL_PATH}")

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    return preprocessor, model


def predict(data: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
    """
    Predict Mental Health Risk for one or multiple student records.
    Input: DataFrame or dict matching feature columns.
    Output: DataFrame with predictions and class probabilities.
    """
    preprocessor, model = load_artifacts()

    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    if 'Student_ID' in df.columns:
        df = df.drop(columns=['Student_ID'])
    if 'Mental_Health_Risk' in df.columns:
        df = df.drop(columns=['Mental_Health_Risk'])

    # Transform features and maintain feature names DataFrame
    X_processed = preprocessor.transform(df)
    feature_names = preprocessor.get_feature_names_out()
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=df.index)

    # Predict
    preds = model.predict(X_processed_df)
    probas = model.predict_proba(X_processed_df)

    results = df.copy()
    results['Predicted_Risk_Code'] = preds
    results['Predicted_Risk'] = [TARGET_INV_MAPPING[p] for p in preds]
    results['Prob_Low'] = [round(p[0], 4) for p in probas]
    results['Prob_Medium'] = [round(p[1], 4) for p in probas]
    results['Prob_High'] = [round(p[2], 4) for p in probas]

    return results


if __name__ == "__main__":
    sample_student = {
        'Age': 21,
        'Gender': 'Female',
        'Year_of_Study': 3,
        'Sleep_Hours': 5.5,
        'Study_Hours_Per_Week': 28.0,
        'GPA': 2.3,
        'Social_Media_Hours': 7.0,
        'Screen_Time_Hours': 10.0,
        'Physical_Activity': 'Low',
        'Financial_Stress': 8.0,
        'Family_Support': 4.0,
        'Attendance_Rate': 75.0,
        'Part_Time_Job': 'Yes',
        'Diet_Quality': 'Poor',
        'Previous_Mental_Health_History': 'Yes'
    }

    res = predict(sample_student)
    print("Sample Prediction:")
    print(res[['Predicted_Risk', 'Prob_Low', 'Prob_Medium', 'Prob_High']])
