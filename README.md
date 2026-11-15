# 🧠 MindPulse-AI: Student Mental Health Risk Prediction & MLOps Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-2.9+-teal.svg)](https://airflow.apache.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**MindPulse-AI** is an end-to-end Machine Learning and MLOps solution designed to assess, predict, and monitor student mental health risks (`Low`, `Medium`, `High`). By analyzing lifestyle, academic, and psychosocial factors, MindPulse-AI provides actionable insights, automated pipeline orchestration, and an interactive decision-support dashboard for educators and counselors.

---

## 📌 Table of Contents

- [Overview & Objectives](#-overview--objectives)
- [Repository Structure](#-repository-structure)
- [Machine Learning Workflow](#-machine-learning-workflow)
  - [1. Data Audit & EDA](#1-data-audit--eda)
  - [2. Feature Engineering & Preprocessing](#2-feature-engineering--preprocessing)
  - [3. Model Training & Evaluation](#3-model-training--evaluation)
- [Orchestration with Apache Airflow](#-orchestration-with-apache-airflow)
- [Interactive Streamlit Application](#-interactive-streamlit-application)
- [Installation & Quickstart](#-installation--quickstart)
- [Pipeline Execution](#-pipeline-execution)
- [Docker & Airflow Deployment](#-docker--airflow-deployment)
- [Results & Performance](#-results--performance)

---

## 🎯 Overview & Objectives

Student well-being directly impacts academic performance and retention. MindPulse-AI addresses the challenge of identifying early warning signs by:
- **Auditing and cleaning** behavioral, academic, and psychological student data.
- **Automating ETL and preprocessing** using production-grade Scikit-Learn transformers.
- **Training and tuning** interpretable and robust multi-class classification models.
- **Orchestrating the complete workflow** via scheduled, fault-tolerant **Apache Airflow** DAGs.
- **Delivering predictions** through an interactive, intuitive **Streamlit** dashboard with personalized recommendations and batch processing.

---

## 📂 Repository Structure

```text
MindPulse-AI/
├── airflow/                        # MLOps orchestration
│   ├── dags/
│   │   └── mindpulse_pipeline.py   # Airflow TaskFlow DAG definition
│   └── docker-compose.yaml         # Containerized Airflow (Postgres + LocalExecutor)
├── app/
│   └── streamlit_app.py            # Streamlit dashboard implementation
├── app.py                          # Root Streamlit launcher
├── data/
│   ├── raw/
│   │   └── raw_data.csv            # Original raw student dataset
│   └── processed/                  # Preprocessed splits (X_train, X_test, y_train, y_test)
├── models/
│   ├── best_model.joblib           # Trained and tuned classification model
│   ├── preprocessor.joblib         # Fitted Scikit-Learn ColumnTransformer
│   └── metrics.json                # Test evaluation report and SLA quality gate
├── notebooks/
│   ├── 1_eda_audit.ipynb           # Step 1: Exploratory Data Analysis & Audit
│   ├── 2_data_processing.ipynb     # Step 2: Preprocessing pipeline development
│   └── 3_training_and_evaluation.ipynb # Step 3 & 4: Modeling, tuning & evaluation
├── src/                            # Modular production pipeline code
│   ├── __init__.py
│   ├── data_processing.py          # Data ingestion, transformation & train/test split
│   ├── train.py                    # Model fitting & artifact persistence
│   ├── evaluate.py                 # Evaluation, confusion matrix & quality gate
│   ├── predict.py                  # Single-instance and batch inference utilities
│   └── pipeline.py                 # End-to-end sequential pipeline runner
├── requirements.txt                # Project dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation
```

---

## 🔬 Machine Learning Workflow

### 1. Data Audit & EDA
- **Exploration (`notebooks/1_eda_audit.ipynb`)**: Examined distributions, feature data types, missing records, duplicates, and outliers.
- **Target Analysis**: Investigated the distribution of `Mental_Health_Risk` across classes (`Low`, `Medium`, `High`) ensuring balanced stratification across splits.

### 2. Feature Engineering & Preprocessing
Implemented in `src/data_processing.py` and exported to `models/preprocessor.joblib`:
- **Numeric Features** (`Age`, `GPA`, `Sleep_Hours`, `Study_Hours_Per_Week`, `Social_Media_Hours`, `Screen_Time_Hours`, `Financial_Stress`, `Family_Support`, `Attendance_Rate`, `Year_of_Study`):
  - Imputation: Median strategy
  - Scaling: `StandardScaler`
- **Ordinal Features** (`Physical_Activity`, `Diet_Quality`):
  - Categorical order mapping via `OrdinalEncoder`
- **Nominal Features** (`Gender`, `Part_Time_Job`, `Previous_Mental_Health_History`):
  - Encoded with `OneHotEncoder(drop='first', handle_unknown='ignore')`

### 3. Model Training & Evaluation
Implemented in `src/train.py` and `src/evaluate.py`:
- **Algorithm**: Tuned `LogisticRegression` with `saga` solver and $L_1$ regularization penalty ($C = 1.0$, `max_iter=1000`).
- **SLA Quality Gate**: Automatically asserts whether Macro F1-score $\ge 0.70$ before accepting model artifacts for deployment.

---

## ⚡ Orchestration with Apache Airflow

The end-to-end workflow is orchestrated using an Airflow DAG (`mindpulse_ml_pipeline`) located in `airflow/dags/mindpulse_pipeline.py`.

```
[ prepare_data ] ──> [ preprocess_data ] ──> [ train_model ] ──> [ evaluate_model ]
```

### DAG Highlights:
- **TaskFlow API**: Clean task definition using `@task` and `@dag` decorators.
- **Failure Alerting**: Custom `on_failure_alert` callback for logging critical task errors.
- **Fault Tolerance**: Automatic task retries (`retries=2`, `retry_delay=3m`).
- **Data & Model Quality Gates**: Raises an exception if input data is malformed or if model F1-score falls below 70%.
- **Schedule**: Set to `@weekly` for continuous periodic retraining.

---

## 💻 Interactive Streamlit Application

The Streamlit web application (`app/streamlit_app.py` or `app.py`) provides a comprehensive interface:

1. **Model Performance Tab**:
   - Interactive KPI cards (Accuracy, Macro Precision, Macro Recall, Macro F1).
   - Seaborn Confusion Matrix heatmap.
   - Full classification report by class.
   - Feature coefficient bar chart explaining model decisions.
2. **Student Risk Predictor Tab**:
   - Intuitive form to input student lifestyle and academic metrics.
   - Instant multi-class probability breakdown and risk categorization.
   - Actionable personalized recommendations based on specific lifestyle indicators (sleep, screen time, study habits, physical activity).
3. **Batch Prediction Tab**:
   - Drag-and-drop CSV upload for batch inference.
   - Real-time prediction and preview with one-click CSV export.

---

## 🚀 Installation & Quickstart

### Prerequisites
- Python 3.10+ (or Python 3.11/3.12)
- Docker & Docker Compose (optional, for Airflow orchestration)

### 1. Clone the Repository
```bash
git clone https://github.com/lioubiarabi/MindPulse-AI.git
cd MindPulse-AI
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Pipeline Execution

You can run the modules individually or trigger the complete pipeline in one command:

```bash
# Run the complete end-to-end ML pipeline
python -m src.pipeline

# Or execute step-by-step:
python -m src.data_processing
python -m src.train
python -m src.evaluate
```

To launch the Streamlit dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🐳 Docker & Airflow Deployment

To launch the containerized Airflow environment:

```bash
# Navigate to airflow folder
cd airflow

# Initialize Airflow environment (first run)
docker compose up airflow-init

# Start Airflow Webserver and Scheduler
docker compose up -d
```

- Access the Airflow UI at: `http://localhost:8080` (Default credentials: `airflow` / `airflow`).
- The `mindpulse_ml_pipeline` DAG will automatically appear and can be triggered manually or run on its weekly schedule.

---

## 📊 Results & Performance

Evaluation conducted on a held-out test set ($20\%$, stratified):

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **77.00%** |
| **Precision (Macro)** | **77.15%** |
| **Recall (Macro)** | **76.92%** |
| **F1-Score (Macro)** | **77.01%** |
| **Quality Gate ($\ge 0.70$)** | **PASSED** |

### Per-Class Breakdown:
- **Low Risk**: Precision: 82.5% | Recall: 78.8% | F1: 80.6%
- **Medium Risk**: Precision: 65.0% | Recall: 67.4% | F1: 66.2%
- **High Risk**: Precision: 83.9% | Recall: 84.6% | F1: 84.2%

---

## 👤 Author & Maintainer

- **Arabi Lioubi** - [GitHub Profile](https://github.com/lioubiarabi)
- **Repository**: [MindPulse-AI](https://github.com/lioubiarabi/MindPulse-AI)
