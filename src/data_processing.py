from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Define project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "raw_data.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

# Feature definitions matching notebook 2
NUMERIC_FEATURES = [
    'Age',
    'Year_of_Study',
    'Sleep_Hours',
    'Study_Hours_Per_Week',
    'GPA',
    'Social_Media_Hours',
    'Screen_Time_Hours',
    'Financial_Stress',
    'Family_Support',
    'Attendance_Rate',
]

ORDINAL_FEATURES = ['Physical_Activity', 'Diet_Quality']
ORDINAL_CATEGORIES = [['Low', 'Moderate', 'High'], ['Poor', 'Average', 'Good']]

NOMINAL_FEATURES = [
    'Gender',
    'Part_Time_Job',
    'Previous_Mental_Health_History',
]

TARGET_COLUMN = 'Mental_Health_Risk'
TARGET_MAPPING = {'Low': 0, 'Medium': 1, 'High': 2}
TARGET_INV_MAPPING = {0: 'Low', 1: 'Medium', 2: 'High'}


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer pipeline matching 2_data_processing.ipynb."""
    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )

    ordinal_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(categories=ORDINAL_CATEGORIES)),
        ]
    )

    nominal_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            (
                'encoder',
                OneHotEncoder(
                    drop='first', sparse_output=False, handle_unknown='ignore'
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('ord', ordinal_transformer, ORDINAL_FEATURES),
            ('nom', nominal_transformer, NOMINAL_FEATURES),
        ]
    )
    return preprocessor


def prepare_raw_data(data_path: Path = RAW_DATA_PATH):
    """Load raw dataset, drop identifier, and split features/target."""
    df = pd.read_csv(data_path)
    if 'Student_ID' in df.columns:
        df = df.drop(columns=['Student_ID'])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map(TARGET_MAPPING)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def process_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    preprocessor_path: Path = PREPROCESSOR_PATH,
    fit_new: bool = False
):
    """
    Transform train and test sets using the preprocessor.
    Uses existing preprocessor.joblib if available, or fits a new one.
    """
    if preprocessor_path.exists() and not fit_new:
        preprocessor = joblib.load(preprocessor_path)
    else:
        preprocessor = build_preprocessor()
        preprocessor.fit(X_train)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, preprocessor_path)

    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(
        X_train_proc, columns=feature_names, index=X_train.index
    )
    X_test_df = pd.DataFrame(
        X_test_proc, columns=feature_names, index=X_test.index
    )

    return X_train_df, X_test_df, preprocessor


def save_processed_data(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path = PROCESSED_DATA_DIR
):
    """Save processed DataFrames and target series into data/processed/."""
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train_df.to_csv(output_dir / "X_train.csv", index=False)
    X_test_df.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print(f"Processed data successfully saved to: {output_dir}")
    print(f"Train shape: {X_train_df.shape} | Nulls: {X_train_df.isnull().sum().sum()}")
    print(f"Test shape:  {X_test_df.shape}  | Nulls: {X_test_df.isnull().sum().sum()}")


def run_data_processing_pipeline():
    """Execute complete data preparation and preprocessing pipeline."""
    print("--- Step 1: Loading raw data and splitting ---")
    X_train, X_test, y_train, y_test = prepare_raw_data()

    print("--- Step 2: Preprocessing features ---")
    X_train_df, X_test_df, _ = process_data(X_train, X_test)

    print("--- Step 3: Saving processed datasets ---")
    save_processed_data(X_train_df, X_test_df, y_train, y_test)


if __name__ == "__main__":
    run_data_processing_pipeline()
