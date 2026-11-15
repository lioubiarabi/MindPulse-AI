import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

TARGET_NAMES = ['Low', 'Medium', 'High']


def evaluate_model(
    processed_dir: Path = PROCESSED_DATA_DIR,
    model_path: Path = BEST_MODEL_PATH,
    metrics_output_path: Path = METRICS_PATH,
    min_f1_threshold: float = 0.70
) -> dict:
    """Evaluate the trained model on test data and save comprehensive metrics."""
    X_test_path = processed_dir / "X_test.csv"
    y_test_path = processed_dir / "y_test.csv"

    if not X_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(f"Processed test data not found in {processed_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test).tolist()

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average='macro'))
    rec = float(recall_score(y_test, y_pred, average='macro'))
    f1 = float(f1_score(y_test, y_pred, average='macro'))

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred, target_names=TARGET_NAMES, output_dict=True
    )

    metrics = {
        "model_type": type(model).__name__,
        "accuracy": round(acc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1, 4),
        "target_names": TARGET_NAMES,
        "confusion_matrix": cm,
        "classification_report": report,
        "passed_threshold": f1 >= min_f1_threshold,
        "min_f1_threshold": min_f1_threshold
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("--- Model Evaluation Summary ---")
    print(f"Test Accuracy:    {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro):    {rec:.4f}")
    print(f"F1-Score (macro):  {f1:.4f}")
    print(f"Threshold Met (>={min_f1_threshold}): {metrics['passed_threshold']}")
    print(f"Evaluation report saved to: {metrics_output_path}")

    return metrics


if __name__ == "__main__":
    evaluate_model()
