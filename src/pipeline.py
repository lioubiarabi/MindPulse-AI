"""
MindPulse-AI Full Pipeline Runner
Executes data preparation, preprocessing, training, and evaluation sequentially.
"""

from pathlib import Path
from src.data_processing import run_data_processing_pipeline
from src.train import train_model
from src.evaluate import evaluate_model


def run_full_pipeline():
    print("=" * 60)
    print("Starting MindPulse-AI ML Pipeline")
    print("=" * 60)

    print("\n>>> Phase 1: Data Preparation & Preprocessing")
    run_data_processing_pipeline()

    print("\n>>> Phase 2: Model Training")
    model = train_model()

    print("\n>>> Phase 3: Model Evaluation")
    metrics = evaluate_model()

    print("\n" + "=" * 60)
    print("Pipeline Execution Completed Successfully!")
    print(f"Final Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Final Test F1-Score: {metrics['f1_macro']:.4f}")
    print("=" * 60)
    return metrics


if __name__ == "__main__":
    run_full_pipeline()
