from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"


def train_model(
    processed_dir: Path = PROCESSED_DATA_DIR,
    model_output_path: Path = BEST_MODEL_PATH
) -> LogisticRegression:
    """Train the best model (tuned Logistic Regression) using processed data."""
    X_train_path = processed_dir / "X_train.csv"
    y_train_path = processed_dir / "y_train.csv"

    if not X_train_path.exists() or not y_train_path.exists():
        raise FileNotFoundError(
            f"Processed data not found in {processed_dir}. Run data_processing first."
        )

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    # Best hyperparameters found in 3_training_and_evaluation.ipynb
    model = LogisticRegression(
        C=1,
        l1_ratio=1,
        max_iter=1000,
        random_state=42,
        solver='saga'
    )

    print("Fitting best model (LogisticRegression with SAGA solver, C=1, l1_ratio=1)...")
    model.fit(X_train, y_train)

    # 5-fold CV score for validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
    print(f"5-Fold CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model saved successfully to: {model_output_path}")

    return model


if __name__ == "__main__":
    train_model()
