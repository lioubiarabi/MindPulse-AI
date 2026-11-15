import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ----------------- CONFIGURATION -----------------
st.set_page_config(
    page_title="MindPulse-AI | Mental Health Risk Predictor",
    page_icon="🧠",
    layout="wide"
)

# ----------------- PATHS & CONSTANTS -----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

TARGET_CLASSES = ['Low', 'Medium', 'High']
RISK_COLORS = {
    'Low': '#2ca02c',
    'Medium': '#ff7f0e',
    'High': '#d62728'
}


# ----------------- LOAD ARTIFACTS -----------------
@st.cache_resource
def load_model_and_preprocessor():
    if not PREPROCESSOR_PATH.exists() or not BEST_MODEL_PATH.exists():
        st.error("Les fichiers de modèles sont introuvables. Exécutez le pipeline d'abord.")
        st.stop()
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    return preprocessor, model


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


preprocessor, model = load_model_and_preprocessor()
metrics_data = load_metrics()


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("🧠 MindPulse-AI")
    st.markdown("**Plateforme d'évaluation & prédiction du risque de santé mentale étudiante**")
    st.divider()

    menu = st.radio(
        "Navigation",
        [
            "📊 Performances du Modèle",
            "🔮 Tester sur un Étudiant",
            "📁 Prédictions en Lot (CSV)",
            "⚙️ Architecture & Pipeline"
        ]
    )

    st.divider()
    st.caption("Projet IA & MLOps • Orchestration Airflow • Modèle Régression Logistique (SAGA)")


# =========================================================================
# PAGE 1: PERFORMANCES DU MODÈLE
# =========================================================================
if menu == "📊 Performances du Modèle":
    st.title("📊 Performances et Évaluation du Modèle")
    st.markdown(
        "Consultez les métriques d'évaluation, la matrice de confusion et l'importance des variables "
        "obtenues sur le jeu de test indépendant (**400 étudiants**)."
    )

    if metrics_data:
        # KPI Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Exactitude (Accuracy)", f"{metrics_data['accuracy'] * 100:.2f}%")
        col2.metric("Précision Macro", f"{metrics_data['precision_macro'] * 100:.2f}%")
        col3.metric("Rappel Macro", f"{metrics_data['recall_macro'] * 100:.2f}%")
        col4.metric("F1-Score Macro", f"{metrics_data['f1_macro'] * 100:.2f}%")

        st.divider()

        tab1, tab2, tab3 = st.tabs(["Matrice de Confusion", "Rapport de Classification Détaillé", "Importance des Variables"])

        with tab1:
            st.subheader("Matrice de Confusion (Jeu de Test)")
            cm = np.array(metrics_data['confusion_matrix'])
            labels = metrics_data['target_names']

            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                cbar=False,
                ax=ax
            )
            ax.set_xlabel("Classe Prédite", fontweight="bold")
            ax.set_ylabel("Classe Réelle", fontweight="bold")
            ax.set_title("Matrice de Confusion (Test Split)", pad=12, fontweight="bold")
            st.pyplot(fig)
            plt.close(fig)

        with tab2:
            st.subheader("Rapport de Classification par Niveau de Risque")
            report = metrics_data['classification_report']
            report_df = pd.DataFrame({
                "Précision": [report['Low']['precision'], report['Medium']['precision'], report['High']['precision']],
                "Rappel (Recall)": [report['Low']['recall'], report['Medium']['recall'], report['High']['recall']],
                "F1-Score": [report['Low']['f1-score'], report['Medium']['f1-score'], report['High']['f1-score']],
                "Effectif (Support)": [int(report['Low']['support']), int(report['Medium']['support']), int(report['High']['support'])]
            }, index=["Faible (Low)", "Modéré (Medium)", "Élevé (High)"])

            st.dataframe(
                report_df.style.format({
                    "Précision": "{:.2%}",
                    "Rappel (Recall)": "{:.2%}",
                    "F1-Score": "{:.2%}",
                    "Effectif (Support)": "{:d}"
                }),
                use_container_width=True
            )

        with tab3:
            st.subheader("Poids des Variables (Coefficients de Régression Logistique)")
            feature_names = preprocessor.get_feature_names_out()
            if hasattr(model, 'coef_'):
                coef_high = model.coef_[2]  # Coefficients for 'High' risk
                clean_names = [f.replace('num__', '').replace('ord__', '').replace('nom__', '') for f in feature_names]
                coef_df = pd.DataFrame({
                    "Variable": clean_names,
                    "Coefficient": coef_high
                }).sort_values(by="Coefficient", key=abs, ascending=True)

                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#d62728' if c > 0 else '#2ca02c' for c in coef_df['Coefficient']]
                ax.barh(coef_df['Variable'], coef_df['Coefficient'], color=colors)
                ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
                ax.set_title("Facteurs favorisant le Risque Élevé (Rouge) vs Protection (Vert)", pad=12, fontweight="bold")
                ax.set_xlabel("Impact Relatif (Coefficient)", fontweight="bold")
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.info("Aucun rapport d'évaluation trouvé. Exécutez `python -m src.evaluate` pour générer les métriques.")


# =========================================================================
# PAGE 2: TESTER SUR UN ÉTUDIANT (INFERENCE)
# =========================================================================
elif menu == "🔮 Tester sur un Étudiant":
    st.title("🔮 Évaluation Individuelle du Risque Étudiant")
    st.markdown("Remplissez le formulaire ci-dessous avec les informations d'un étudiant pour obtenir une prédiction immédiate du niveau de risque.")

    with st.form("student_form"):
        st.subheader("1. Informations Académiques & Personnelles")
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            age = st.number_input("Âge", min_value=16, max_value=40, value=21, step=1)
            gender = st.selectbox("Genre", ["Male", "Female", "Non-binary"])
        with col_a2:
            year_of_study = st.slider("Année d'études (Year of Study)", min_value=1, max_value=5, value=2)
            part_time_job = st.selectbox("Emploi étudiant (Part-time Job)", ["No", "Yes"])
        with col_a3:
            gpa = st.number_input("GPA (Moyenne académique)", min_value=0.0, max_value=4.0, value=2.6, step=0.05)
            attendance_rate = st.slider("Taux de présence (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)

        st.subheader("2. Habitudes de Vie & Temps d'Écran")
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            sleep_hours = st.number_input("Heures de sommeil par jour", min_value=2.0, max_value=14.0, value=6.5, step=0.5)
            study_hours = st.number_input("Heures d'étude par semaine", min_value=0.0, max_value=70.0, value=20.0, step=1.0)
        with col_b2:
            screen_time = st.number_input("Temps d'écran quotidien (heures)", min_value=1.0, max_value=18.0, value=8.5, step=0.5)
            social_media = st.number_input("Réseaux sociaux par jour (heures)", min_value=0.0, max_value=15.0, value=5.5, step=0.5)
        with col_b3:
            physical_activity = st.selectbox("Activité physique", ["Low", "Moderate", "High"], index=1)
            diet_quality = st.selectbox("Qualité de l'alimentation", ["Poor", "Average", "Good"], index=1)

        st.subheader("3. Facteurs Psychosociaux")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            financial_stress = st.slider("Stress financier (1: Faible, 10: Élevé)", min_value=1.0, max_value=10.0, value=6.0, step=0.5)
        with col_c2:
            family_support = st.slider("Soutien familial (1: Faible, 10: Élevé)", min_value=1.0, max_value=10.0, value=6.5, step=0.5)
        with col_c3:
            prev_history = st.selectbox("Antécédents de santé mentale", ["No", "Yes"])

        submitted = st.form_submit_button("🚀 Prédire le Risque de Santé Mentale", type="primary", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Year_of_Study': year_of_study,
            'Sleep_Hours': sleep_hours,
            'Study_Hours_Per_Week': study_hours,
            'GPA': gpa,
            'Social_Media_Hours': social_media,
            'Screen_Time_Hours': screen_time,
            'Physical_Activity': physical_activity,
            'Financial_Stress': financial_stress,
            'Family_Support': family_support,
            'Attendance_Rate': attendance_rate,
            'Part_Time_Job': part_time_job,
            'Diet_Quality': diet_quality,
            'Previous_Mental_Health_History': prev_history
        }])

        # Transform using preprocessor
        X_proc = preprocessor.transform(input_data)
        feature_names = preprocessor.get_feature_names_out()
        X_proc_df = pd.DataFrame(X_proc, columns=feature_names)

        pred_code = model.predict(X_proc_df)[0]
        probas = model.predict_proba(X_proc_df)[0]
        pred_label = TARGET_CLASSES[pred_code]

        st.divider()
        st.subheader("Résultat du Diagnostic IA")

        res_col1, res_col2 = st.columns([1, 1.5])
        with res_col1:
            badge_color = RISK_COLORS[pred_label]
            st.markdown(
                f"""
                <div style="background-color: {badge_color}20; border-left: 6px solid {badge_color}; padding: 18px; border-radius: 8px;">
                    <h3 style="color: {badge_color}; margin: 0;">Risque Prédit : {pred_label}</h3>
                    <p style="margin: 8px 0 0 0; color: #333; font-size: 16px;">
                        Probabilité dominante : <b>{probas[pred_code] * 100:.1f}%</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")
            if pred_label == "High":
                st.error("⚠️ **Recommandation :** Risque élevé détecté. Prévoir un entretien de soutien psychologique et un aménagement des charges académiques.")
            elif pred_label == "Medium":
                st.warning("⚡ **Recommandation :** Risque modéré. Encourager l'hygiène du sommeil, la régulation du temps d'écran et des ateliers de gestion du stress.")
            else:
                st.success("✅ **Recommandation :** Profil équilibré. Poursuivre les bonnes habitudes de vie et le suivi régulier.")

        with res_col2:
            st.markdown("##### Distribution des Probabilités")
            prob_df = pd.DataFrame({
                "Niveau de Risque": TARGET_CLASSES,
                "Probabilité (%)": [p * 100 for p in probas]
            })

            fig, ax = plt.subplots(figsize=(6, 2.8))
            bars = ax.barh(prob_df["Niveau de Risque"], prob_df["Probabilité (%)"], color=[RISK_COLORS[c] for c in TARGET_CLASSES])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probabilité (%)")
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va='center', fontweight='bold')
            sns.despine(top=True, right=True)
            st.pyplot(fig)
            plt.close(fig)


# =========================================================================
# PAGE 3: PRÉDICTIONS EN LOT (CSV)
# =========================================================================
elif menu == "📁 Prédictions en Lot (CSV)":
    st.title("📁 Prédictions en Lot (Batch Testing)")
    st.markdown("Importez un fichier CSV contenant les données de plusieurs étudiants pour exécuter le pipeline d'inférence en masse.")

    uploaded_file = st.file_uploader("Importer un fichier CSV (structure brute ou test)", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"**{len(batch_df)} enregistrements importés.** Aperçu :")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("⚡ Lancer les prédictions sur tout le fichier", type="primary"):
                with st.spinner("Prétraitement et inférence en cours..."):
                    from src.predict import predict
                    results = predict(batch_df)

                st.success("Inférence terminée avec succès !")

                # Risk counts
                st.subheader("Répartition des Prédictions")
                counts = results['Predicted_Risk'].value_counts()
                c1, c2, c3 = st.columns(3)
                c1.metric("Risque Faible (Low)", counts.get("Low", 0))
                c2.metric("Risque Modéré (Medium)", counts.get("Medium", 0))
                c3.metric("Risque Élevé (High)", counts.get("High", 0))

                st.dataframe(results[['Predicted_Risk', 'Prob_Low', 'Prob_Medium', 'Prob_High'] + [c for c in results.columns if c not in ['Predicted_Risk', 'Prob_Low', 'Prob_Medium', 'Prob_High', 'Predicted_Risk_Code']]], use_container_width=True)

                # Download CSV
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger les résultats avec prédictions (CSV)",
                    data=csv,
                    file_name="mindpulse_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Erreur lors de la lecture ou prédiction : {e}")
    else:
        st.info("💡 Vous pouvez tester avec `data/raw/raw_data.csv` pour évaluer l'ensemble du dataset.")


# =========================================================================
# PAGE 4: ARCHITECTURE & PIPELINE
# =========================================================================
elif menu == "⚙️ Architecture & Pipeline":
    st.title("⚙️ Architecture du Pipeline & Orchestration")
    st.markdown(
        """
        ### Pipeline MLOps MindPulse-AI
        Cette architecture automatise le cycle de vie du modèle de détection du risque de santé mentale :
        """
    )

    st.markdown(
        """
        ```
        [ data/raw/raw_data.csv ]
                   │
                   ▼
        [ src/data_processing.py ]  ──▶ Utilise models/preprocessor.joblib 
                   │                ──▶ Génère data/processed/ (X_train, X_test, y_train, y_test)
                   ▼
        [ src/train.py ]            ──▶ Entraîne LogisticRegression (SAGA, C=1, l1=1)
                   │                ──▶ Sauvegarde models/best_model.joblib
                   ▼
        [ src/evaluate.py ]         ──▶ Évalue les métriques (Accuracy, F1-macro >= 0.70)
                   │                ──▶ Génère models/metrics.json
                   ▼
        [ Airflow DAG ]             ──▶ Orchestre et planifie l'exécution hebdomadaire
                   │
                   ▼
        [ Streamlit App ]           ──▶ Consultation interactive des métriques & Inférence
        ```
        """
    )

    st.divider()
    st.subheader("Statut des Fichiers Clés")
    files_status = [
        ("Données brutes", DATA_DIR / "raw" / "raw_data.csv"),
        ("Données prétraitées (X_train)", DATA_DIR / "processed" / "X_train.csv"),
        ("Transformeur entraîné", PREPROCESSOR_PATH),
        ("Meilleur Modèle", BEST_MODEL_PATH),
        ("Rapport d'évaluation", METRICS_PATH),
        ("DAG Airflow", PROJECT_ROOT / "airflow" / "dags" / "mindpulse_pipeline.py"),
    ]

    for label, path in files_status:
        if path.exists():
            st.markdown(f"✅ **{label}** : `{path.relative_to(PROJECT_ROOT)}` ({path.stat().st_size:,} octets)")
        else:
            st.markdown(f"❌ **{label}** : `{path.relative_to(PROJECT_ROOT)}` (Introuvable)")
