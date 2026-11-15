"""
Airflow DAG: MindPulse-AI End-to-End Machine Learning Pipeline
Orchestrates:
1. prepare_data: Validates raw dataset existence, integrity, and schema.
2. preprocess_data: Preprocesses features using preprocessor.joblib and saves data/processed/ files.
3. train_model: Trains tuned Logistic Regression model and saves best_model.joblib.
4. evaluate_model: Evaluates test set performance, verifies quality gate (F1 >= 0.70), and saves metrics.json.
"""

from datetime import datetime, timedelta
import json
import logging
import os
import sys
from pathlib import Path

# Support running both inside Docker (/opt/mindpulse) and locally on Host
if Path("/opt/mindpulse").exists():
    PROJECT_ROOT = Path("/opt/mindpulse")
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airflow.decorators import dag, task

logger = logging.getLogger(__name__)


def on_failure_alert(context):
    """Callback triggered if any task fails to log details and notify."""
    task_instance = context.get('task_instance')
    dag_id = context.get('dag').dag_id
    exec_date = context.get('execution_date')
    exception = context.get('exception')
    logger.error(
        f"🚨 AIRFLOW FAILURE ALERT | DAG: {dag_id} | Task: {task_instance.task_id} | "
        f"Execution Date: {exec_date} | Error: {exception}"
    )


default_args = {
    'owner': 'MindPulse-AI',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
    'on_failure_callback': on_failure_alert,
}


@dag(
    dag_id='mindpulse_ml_pipeline',
    default_args=default_args,
    description='Automated pipeline for student mental health risk prediction (ETL, training, evaluation)',
    schedule='@weekly',
    start_date=datetime(2026, 11, 10),
    catchup=False,
    tags=['MindPulse', 'MLOps', 'MentalHealth', 'Classification'],
)
def mindpulse_pipeline():

    @task(task_id='prepare_data')
    def prepare_data_task():
        """Verify raw data integrity and schema."""
        import pandas as pd
        from src.data_processing import RAW_DATA_PATH, TARGET_COLUMN

        logger.info(f"Checking raw data file at: {RAW_DATA_PATH}")
        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError(f"Missing raw data file: {RAW_DATA_PATH}")

        df = pd.read_csv(RAW_DATA_PATH)
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' missing from raw data!")

        row_count = len(df)
        col_count = len(df.columns)
        logger.info(f"Raw data validated: {row_count} rows, {col_count} columns.")

        return {"row_count": row_count, "col_count": col_count}

    @task(task_id='preprocess_data')
    def preprocess_data_task(prep_info: dict):
        """Run feature preprocessing and generate data/processed files."""
        from src.data_processing import (
            prepare_raw_data,
            process_data,
            save_processed_data,
            PROCESSED_DATA_DIR
        )

        logger.info(f"Processing data with input row count: {prep_info.get('row_count')}")
        X_train, X_test, y_train, y_test = prepare_raw_data()
        X_train_df, X_test_df, _ = process_data(X_train, X_test)
        save_processed_data(X_train_df, X_test_df, y_train, y_test)

        logger.info(f"Saved processed datasets to: {PROCESSED_DATA_DIR}")
        return {
            "train_samples": len(X_train_df),
            "test_samples": len(X_test_df),
            "features_count": X_train_df.shape[1]
        }

    @task(task_id='train_model')
    def train_model_task(proc_info: dict):
        """Train tuned classification model and persist best_model.joblib."""
        from src.train import train_model, BEST_MODEL_PATH

        logger.info(f"Training on {proc_info.get('train_samples')} processed samples.")
        model = train_model()
        logger.info(f"Model {type(model).__name__} saved to {BEST_MODEL_PATH}.")
        return {"model_type": type(model).__name__, "status": "trained"}

    @task(task_id='evaluate_model')
    def evaluate_model_task(train_info: dict):
        """Evaluate model against SLA quality gate and save performance report."""
        from src.evaluate import evaluate_model, METRICS_PATH

        logger.info("Evaluating model on test set...")
        metrics = evaluate_model(min_f1_threshold=0.70)

        f1 = metrics["f1_macro"]
        acc = metrics["accuracy"]
        logger.info(f"Evaluation metrics: Accuracy={acc}, F1-Macro={f1}")

        if not metrics["passed_threshold"]:
            raise ValueError(
                f"Model quality gate failed! Macro F1 ({f1:.4f}) is below threshold ({metrics['min_f1_threshold']})."
            )

        logger.info(f"Model passed quality gate! Report saved to {METRICS_PATH}")
        return {
            "accuracy": acc,
            "f1_macro": f1,
            "quality_gate": "PASSED"
        }

    # Pipeline task dependencies
    raw_info = prepare_data_task()
    proc_info = preprocess_data_task(raw_info)
    model_info = train_model_task(proc_info)
    eval_info = evaluate_model_task(model_info)


dag_instance = mindpulse_pipeline()
