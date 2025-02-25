import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dataset.state import DatasetState
from dataset.forms.config import dataset_config_form
import utils.training as train  # Suppose que train_and_evaluate_model(model) retourne (model, y_pred, cv_scores, report, conf_matrix)
from utils.model_storage import ModelStorage

# ---------------------------------------------------------------------
# Style CSS personnalisé
# ---------------------------------------------------------------------
st.markdown("""
    <style>
        .section-header {
            background: linear-gradient(135deg, #6a11cb, #2575fc);
            padding: 1.5rem;
            border-radius: 8px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            font-family: 'Arial', sans-serif;
        }
        .info-box {
            background-color: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            color: #FFFFFF;
            font-family: 'Arial', sans-serif;
        }
        .info-box h4 {
            margin-bottom: 0.5rem;
        }
        .metrics-container {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# En-tête de la page
# ---------------------------------------------------------------------
st.markdown('<div class="section-header"><h1>🔍 Comparaison des Modèles</h1></div>', unsafe_allow_html=True)
st.markdown("""
Cette page compare les différents modèles selon plusieurs métriques.  
Chaque onglet propose un graphique comparatif pour la métrique choisie et un commentaire 
pour indiquer le meilleur modèle.
""")

# ---------------------------------------------------------------------
# Vérification du dataset
# ---------------------------------------------------------------------
dataset = DatasetState.get_dataset()
if dataset is None:
    st.markdown("### :ghost: Aucun jeu de données")
    st.caption("Chargez un jeu de données pour pouvoir effectuer la comparaison.")
    st.button("📤 Charger un Dataset", type="primary", on_click=dataset_config_form, use_container_width=True)
    st.stop()

# Vérification si des modèles ont été entraînés
if not ModelStorage.has_trained_models():
    st.warning("⚠️ Aucun modèle n'a été entraîné. Veuillez d'abord entraîner des modèles dans la page Entraînement.")
    st.info("👈 Allez à la page 'Entraînement' pour entraîner des modèles.")
    st.stop()

# Récupération des résultats des modèles entraînés
results_dict = ModelStorage.get_model_results()

# Création des DataFrames pour la comparaison
accuracy_data = []
precision_data = []
recall_data = []
f1_data = []
cv_data = []

for model_name, results in results_dict.items():
    accuracy_data.append({"Modèle": model_name, "Valeur": results["accuracy"]})
    precision_data.append({"Modèle": model_name, "Valeur": results["precision"]})
    recall_data.append({"Modèle": model_name, "Valeur": results["recall"]})
    f1_data.append({"Modèle": model_name, "Valeur": results["f1"]})
    cv_data.append({"Modèle": model_name, "Valeur": results["cv_scores"].mean()})

df_accuracy = pd.DataFrame(accuracy_data)
df_precision = pd.DataFrame(precision_data)
df_recall = pd.DataFrame(recall_data)
df_f1 = pd.DataFrame(f1_data)
df_cv = pd.DataFrame(cv_data)

# Fonction pour créer un graphique de comparaison
def create_comparison_chart(df, title, y_label):
    fig = px.bar(
        df, 
        x="Modèle", 
        y="Valeur", 
        color="Modèle",
        title=title,
        labels={"Valeur": y_label}
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.05)'
    )
    return fig

# Création des onglets pour chaque métrique
tab_acc, tab_prec, tab_rec, tab_f1, tab_cv = st.tabs([
    "Accuracy", 
    "Precision", 
    "Recall", 
    "F1 Score", 
    "Cross-Validation"
])

# Onglet Accuracy
with tab_acc:
    st.subheader("Comparaison par Accuracy")
    fig_acc = create_comparison_chart(df_accuracy, "Accuracy des modèles", "Accuracy")
    st.plotly_chart(fig_acc, use_container_width=True)
    
    # Meilleur modèle
    best_acc = df_accuracy["Valeur"].max()
    best_model_acc = df_accuracy.loc[df_accuracy["Valeur"] == best_acc, "Modèle"].iloc[0]
    st.markdown(f"**Le meilleur modèle selon l'Accuracy est {best_model_acc} avec un score de {best_acc:.3f}.**")

# Onglet Precision
with tab_prec:
    st.subheader("Comparaison par Precision")
    fig_prec = create_comparison_chart(df_precision, "Precision des modèles", "Precision")
    st.plotly_chart(fig_prec, use_container_width=True)
    
    # Meilleur modèle
    best_prec = df_precision["Valeur"].max()
    best_model_prec = df_precision.loc[df_precision["Valeur"] == best_prec, "Modèle"].iloc[0]
    st.markdown(f"**Le meilleur modèle selon la Precision est {best_model_prec} avec un score de {best_prec:.3f}.**")

# Onglet Recall
with tab_rec:
    st.subheader("Comparaison par Recall")
    fig_rec = create_comparison_chart(df_recall, "Recall des modèles", "Recall")
    st.plotly_chart(fig_rec, use_container_width=True)
    
    # Meilleur modèle
    best_rec = df_recall["Valeur"].max()
    best_model_rec = df_recall.loc[df_recall["Valeur"] == best_rec, "Modèle"].iloc[0]
    st.markdown(f"**Le meilleur modèle selon le Recall est {best_model_rec} avec un score de {best_rec:.3f}.**")

# Onglet F1 Score
with tab_f1:
    st.subheader("Comparaison par F1 Score")
    fig_f1 = create_comparison_chart(df_f1, "F1 Score des modèles", "F1 Score")
    st.plotly_chart(fig_f1, use_container_width=True)
    
    # Meilleur modèle
    best_f1 = df_f1["Valeur"].max()
    best_model_f1 = df_f1.loc[df_f1["Valeur"] == best_f1, "Modèle"].iloc[0]
    st.markdown(f"**Le meilleur modèle selon le F1 Score est {best_model_f1} avec un score de {best_f1:.3f}.**")

# Onglet Cross-Validation
with tab_cv:
    st.subheader("Comparaison par Cross-Validation")
    fig_cv = create_comparison_chart(df_cv, "Cross-Validation des modèles", "CV Score")
    st.plotly_chart(fig_cv, use_container_width=True)
    
    # Meilleur modèle
    best_cv = df_cv["Valeur"].max()
    best_model_cv = df_cv.loc[df_cv["Valeur"] == best_cv, "Modèle"].iloc[0]
    st.markdown(f"**Le meilleur modèle selon la Cross-Validation est {best_model_cv} avec un score de {best_cv:.3f}.**")

# Tableau récapitulatif
st.subheader("Tableau récapitulatif des performances")
summary_data = []
for model_name, results in results_dict.items():
    summary_data.append({
        "Modèle": model_name,
        "Accuracy": f"{results['accuracy']:.3f}",
        "Precision": f"{results['precision']:.3f}",
        "Recall": f"{results['recall']:.3f}",
        "F1 Score": f"{results['f1']:.3f}",
        "CV Score": f"{results['cv_scores'].mean():.3f}"
    })

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True)

# Recommandation finale
st.markdown("""
    <div class="info-box">
        <h4>📊 Recommandation finale</h4>
        <p>En fonction des différentes métriques, voici notre recommandation :</p>
    </div>
""", unsafe_allow_html=True)

# Calcul du score global (moyenne des rangs pour chaque métrique)
model_ranks = {}
for model in results_dict.keys():
    model_ranks[model] = 0

# Calcul des rangs pour chaque métrique
for df, weight in [(df_accuracy, 1), (df_precision, 1), (df_recall, 1), (df_f1, 1.5), (df_cv, 1.5)]:
    df_sorted = df.sort_values("Valeur", ascending=False)
    for i, model in enumerate(df_sorted["Modèle"]):
        model_ranks[model] += (i + 1) * weight

# Modèle avec le meilleur rang global
best_model = min(model_ranks.items(), key=lambda x: x[1])[0]

st.success(f"**Le modèle recommandé est : {best_model}**")
st.markdown(f"""
    Ce modèle offre le meilleur équilibre entre les différentes métriques d'évaluation.
    Pour une utilisation en production, nous recommandons d'utiliser ce modèle.
""")
