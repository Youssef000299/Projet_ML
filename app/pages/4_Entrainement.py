import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import plotly.express as px
import plotly.graph_objects as go
import math

# ===== Ajouts pour la visualisation des arbres =====
import matplotlib.pyplot as plt
from sklearn import tree

from dataset.state import DatasetState
from dataset.forms.config import dataset_config_form
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
        .model-container {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1.5rem 0;
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
            color: #FFFFFF;
            margin-bottom: 0.5rem;
        }
        .info-box p {
            margin: 0.5rem 0;
            font-size: 0.95rem;
        }
        .info-box ul {
            margin: 0.5rem 0 0.5rem 1.2rem;
            padding: 0;
        }
        .info-box li {
            margin-bottom: 0.4rem;
        }
        .metrics-container {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        /* Variables pour le mode clair/sombre */
        [data-testid="stAppViewContainer"] {
            --primary-color: #722F37;
            --secondary-color: #9A3324;
            --feature-bg: rgba(255, 255, 255, 0.05);
            --border-color: rgba(255, 255, 255, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# En-tête de la page
# ---------------------------------------------------------------------
st.markdown('<div class="section-header"><h1>🤖 Entraînement des Modèles</h1></div>', unsafe_allow_html=True)

# Récupération du dataset (données nettoyées)
dataset = DatasetState.get_dataset()
if dataset is None:
    st.error("Aucun jeu de données n'est disponible. Veuillez d'abord charger des données.")
    st.stop()

# Confirmation explicite que nous utilisons les données nettoyées
st.info("📋 Les modèles seront entraînés sur les données prétraitées dans l'étape de nettoyage.")

# Préparation des données
df = dataset.data.copy()  # Utilisation des données nettoyées
X = df[dataset.features_columns]
y = df[dataset.target_columns[0]]

# Affichage des informations sur les données
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Nombre d'échantillons", f"{len(df)}")
with col2:
    st.metric("Nombre de features", f"{len(dataset.features_columns)}")
with col3:
    st.metric("Classes cibles", f"{len(df[dataset.target_columns[0]].unique())}")

# ---------------------------------------------------------------------
# Fonctions d'entraînement et de visualisation
# ---------------------------------------------------------------------
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    return {
        'model': model,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_scores': cv_scores
    }

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Prédiction", y="Réalité", color="Nombre"),
        title="Matrice de Confusion"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.05)'
    )
    return fig

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig = px.bar(
            x=[feature_names[i] for i in indices],
            y=importances[indices],
            labels={'x': 'Features', 'y': 'Importance'},
            title='Importance des Features'
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.05)',
        )
        return fig
    return None

def plot_cross_validation(cv_scores):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Fold {i+1}" for i in range(len(cv_scores))],
        y=cv_scores,
        marker_color='#4CAF50'
    ))
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=cv_scores.mean(),
        x1=len(cv_scores)-0.5,
        y1=cv_scores.mean(),
        line=dict(color="red", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=len(cv_scores)-1,
        y=cv_scores.mean(),
        text=f"Moyenne: {cv_scores.mean():.3f}",
        showarrow=True,
        arrowhead=1,
    )
    fig.update_layout(
        title="Scores de Cross-Validation",
        xaxis_title="Fold",
        yaxis_title="Accuracy",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.05)',
    )
    return fig

def display_metrics(results):
    st.markdown("""
        <div class="metrics-container">
            <h3>📊 Métriques d'évaluation</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{results['accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{results['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{results['recall']:.3f}")
    with col4:
        st.metric("F1 Score", f"{results['f1']:.3f}")
    
    st.markdown("**Cross-Validation (5-fold)**")
    st.metric("CV Score moyen", f"{results['cv_scores'].mean():.3f} ± {results['cv_scores'].std():.3f}")

# Fonction pour afficher les résultats d'un modèle précédemment entraîné
def display_saved_results(model_name, results):
    st.success(f"✅ Modèle {model_name} déjà entraîné")
    
    # Affichage des métriques
    display_metrics(results)
    
    # Récupération des données pour les visualisations
    y_test = results.get('y_test', [])
    y_pred = results.get('y_pred', [])
    cv_scores = results.get('cv_scores', np.array([]))
    model = results.get('model', None)
    
    # Visualisations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Matrice de Confusion")
        if len(y_test) > 0 and len(y_pred) > 0:
            fig_cm = plot_confusion_matrix(y_test, y_pred)
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Données insuffisantes pour afficher la matrice de confusion")
    
    with col2:
        if model_name in ["Arbre de Décision", "Random Forest"]:
            st.markdown("### Importance des Features")
            if model and hasattr(model, 'feature_importances_'):
                fig_importance = plot_feature_importance(model, dataset.features_columns)
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info("Données insuffisantes pour afficher l'importance des features")
        else:
            st.markdown("### Cross-Validation")
            if len(cv_scores) > 0:
                fig_cv = plot_cross_validation(cv_scores)
                st.plotly_chart(fig_cv, use_container_width=True)
            else:
                st.info("Données insuffisantes pour afficher la cross-validation")
    
    # Cross-validation pour les modèles d'arbres
    if model_name in ["Arbre de Décision", "Random Forest"]:
        st.markdown("### Cross-Validation")
        if len(cv_scores) > 0:
            fig_cv = plot_cross_validation(cv_scores)
            st.plotly_chart(fig_cv, use_container_width=True)
        else:
            st.info("Données insuffisantes pour afficher la cross-validation")
    
    # Classification Report
    st.markdown("### Rapport de Classification")
    if 'classification_report' in results:
        report_df = pd.DataFrame(results['classification_report']).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
    else:
        st.info("Données insuffisantes pour afficher le rapport de classification")
    
    # Visualisation de l'arbre pour les modèles d'arbres
    if model_name == "Arbre de Décision" and model:
        st.markdown("### Visualisation de l'Arbre de Décision")
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            tree.plot_tree(
                model,
                filled=True,
                feature_names=dataset.features_columns,
                class_names=[str(cls) for cls in y.unique()],
                ax=ax
            )
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur lors de la visualisation de l'arbre: {e}")
    
    # Visualisation de quelques arbres pour Random Forest
    if model_name == "Random Forest" and model and hasattr(model, 'estimators_'):
        st.markdown("### Visualisation de quelques Arbres de la Forêt")
        try:
            nb_to_display = min(2, len(model.estimators_))
            for i in range(nb_to_display):
                st.markdown(f"#### Arbre n°{i+1}")
                fig, ax = plt.subplots(figsize=(12, 8))
                tree.plot_tree(
                    model.estimators_[i],
                    filled=True,
                    feature_names=dataset.features_columns,
                    class_names=[str(cls) for cls in y.unique()],
                    ax=ax
                )
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur lors de la visualisation des arbres: {e}")

# ---------------------------------------------------------------------
# Création des 3 onglets
# ---------------------------------------------------------------------
tab_logistic, tab_tree, tab_forest = st.tabs([
    "Régression Logistique",
    "Arbre de Décision",
    "Random Forest"
])

# ============== Onglet Régression Logistique ==============
with tab_logistic:
    st.markdown("""
        <div class="model-container">
            <h3>📈 Régression Logistique</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Vérifier si le modèle a déjà été entraîné
    saved_results = ModelStorage.get_model_results("Régression Logistique")
    
    # Explication du modèle
    st.markdown("""
        <div class="info-box">
            <h4>Qu'est-ce que la Régression Logistique ?</h4>
            <p>Modèle linéaire pour la classification, qui prédit la probabilité d'appartenance à une classe 
               via une fonction logistique (sigmoïde).</p>
            <p><strong>Paramètres :</strong></p>
            <ul>
                <li><strong>C</strong> : Inverse de la régularisation (0.01 à 100). 
                    Plus C est petit, plus la régularisation est forte.</li>
                <li><strong>Solver</strong> : Algorithme d'optimisation (lbfgs, liblinear, newton-cg, sag, saga).</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Si le modèle a déjà été entraîné, afficher les résultats
    if saved_results:
        display_saved_results("Régression Logistique", saved_results)
    else:
        st.session_state.logistic_trained = False
    
    # Afficher les paramètres si nécessaire
    if not st.session_state.get("logistic_trained", False):
        colA, colB = st.columns(2)
        with colA:
            C = st.select_slider("Paramètre de régularisation (C)", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
        with colB:
            solver = st.selectbox("Solver", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"], index=0)
        
        if st.button("Entraîner la Régression Logistique"):
            with st.spinner("Entraînement en cours..."):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)
                results = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
                
                # Ajouter y_test et classification_report aux résultats
                results['y_test'] = y_test
                report = classification_report(y_test, results['y_pred'], output_dict=True)
                results['classification_report'] = report
                
                # Stocker les résultats pour la page de comparaison
                ModelStorage.store_model_results("Régression Logistique", results)
                
                # Afficher les résultats
                display_metrics(results)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Matrice de Confusion")
                    fig_cm = plot_confusion_matrix(y_test, results['y_pred'])
                    st.plotly_chart(fig_cm, use_container_width=True)
                with col2:
                    st.markdown("### Cross-Validation")
                    fig_cv = plot_cross_validation(results['cv_scores'])
                    st.plotly_chart(fig_cv, use_container_width=True)
                
                st.markdown("### Rapport de Classification")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
                
                # Au lieu de cacher les paramètres et faire un rerun,
                # simplement stocker un indicateur dans session_state
                st.session_state.logistic_trained = True

# ============== Onglet Arbre de Décision ==============
with tab_tree:
    st.markdown("""
        <div class="model-container">
            <h3>🌳 Arbre de Décision</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Vérifier si le modèle a déjà été entraîné
    saved_results = ModelStorage.get_model_results("Arbre de Décision")
    
    # Explication du modèle
    st.markdown("""
        <div class="info-box">
            <h4>Qu'est-ce qu'un Arbre de Décision ?</h4>
            <p>Modèle qui segmente les données via des règles successives pour obtenir 
               des prédictions simples et interprétables.</p>
            <p><strong>Paramètres :</strong></p>
            <ul>
                <li><strong>Profondeur maximale</strong> : Limite la profondeur de l'arbre (1 à 20).</li>
                <li><strong>Min samples leaf</strong> : Nombre minimum d'échantillons par feuille (1 à 20).</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Si le modèle a déjà été entraîné, afficher les résultats
    if saved_results:
        display_saved_results("Arbre de Décision", saved_results)
    else:
        st.session_state.tree_trained = False
    
    # Afficher les paramètres si nécessaire
    if not st.session_state.get("tree_trained", False):
        colA, colB = st.columns(2)
        with colA:
            max_depth = st.slider("Profondeur maximale", min_value=1, max_value=20, value=5)
        with colB:
            min_samples_leaf = st.slider("Échantillons min par feuille", min_value=1, max_value=20, value=5)
        
        if st.button("Entraîner l'Arbre de Décision"):
            with st.spinner("Entraînement en cours..."):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
                results = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
                
                # Ajouter y_test et classification_report aux résultats
                results['y_test'] = y_test
                report = classification_report(y_test, results['y_pred'], output_dict=True)
                results['classification_report'] = report
                
                # Stocker les résultats pour la page de comparaison
                ModelStorage.store_model_results("Arbre de Décision", results)
                
                # Afficher les résultats
                display_metrics(results)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Matrice de Confusion")
                    fig_cm = plot_confusion_matrix(y_test, results['y_pred'])
                    st.plotly_chart(fig_cm, use_container_width=True)
                with col2:
                    st.markdown("### Importance des Features")
                    fig_importance = plot_feature_importance(model, dataset.features_columns)
                    if fig_importance:
                        st.plotly_chart(fig_importance, use_container_width=True)
                    else:
                        st.info("Aucun graphique d'importance n'est disponible pour ce modèle.")
                
                st.markdown("### Cross-Validation")
                fig_cv = plot_cross_validation(results['cv_scores'])
                st.plotly_chart(fig_cv, use_container_width=True)
                
                st.markdown("### Rapport de Classification")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
                
                # Visualisation de l'arbre
                st.markdown("### Visualisation de l'Arbre de Décision")
                fig, ax = plt.subplots(figsize=(12, 8))
                tree.plot_tree(
                    model,
                    filled=True,
                    feature_names=dataset.features_columns,
                    class_names=[str(cls) for cls in y.unique()],
                    ax=ax
                )
                st.pyplot(fig)
                
                # Cacher les paramètres après l'entraînement
                st.session_state.tree_trained = True

# ============== Onglet Random Forest ==============
with tab_forest:
    st.markdown("""
        <div class="model-container">
            <h3>🌲 Random Forest</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Vérifier si le modèle a déjà été entraîné
    saved_results = ModelStorage.get_model_results("Random Forest")
    
    # Explication du modèle
    st.markdown("""
        <div class="info-box">
            <h4>Qu'est-ce qu'une Random Forest ?</h4>
            <p>Un ensemble d'arbres de décision entraînés sur des échantillons aléatoires. 
               La prédiction finale est la majorité des votes des arbres.</p>
            <p><strong>Paramètres :</strong></p>
            <ul>
                <li><strong>Nombre d'arbres</strong> (n_estimators) : ex. 50 à 300.</li>
                <li><strong>Profondeur maximale</strong> : ex. 5 à 20.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Si le modèle a déjà été entraîné, afficher les résultats
    if saved_results:
        display_saved_results("Random Forest", saved_results)
    else:
        st.session_state.forest_trained = False
    
    # Afficher les paramètres si nécessaire
    if not st.session_state.get("forest_trained", False):
        colA, colB = st.columns(2)
        with colA:
            n_estimators = st.slider("Nombre d'arbres (n_estimators)", min_value=10, max_value=300, value=100, step=10)
        with colB:
            max_depth_rf = st.slider("Profondeur maximale", min_value=1, max_value=30, value=10)
        
        if st.button("Entraîner la Random Forest"):
            with st.spinner("Entraînement en cours..."):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth_rf, random_state=42)
                results = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
                
                # Ajouter y_test et classification_report aux résultats
                results['y_test'] = y_test
                report = classification_report(y_test, results['y_pred'], output_dict=True)
                results['classification_report'] = report
                
                # Stocker les résultats pour la page de comparaison
                ModelStorage.store_model_results("Random Forest", results)
                
                # Afficher les résultats
                display_metrics(results)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Matrice de Confusion")
                    fig_cm = plot_confusion_matrix(y_test, results['y_pred'])
                    st.plotly_chart(fig_cm, use_container_width=True)
                with col2:
                    st.markdown("### Importance des Features")
                    fig_importance = plot_feature_importance(model, dataset.features_columns)
                    if fig_importance:
                        st.plotly_chart(fig_importance, use_container_width=True)
                    else:
                        st.info("Aucun graphique d'importance n'est disponible pour ce modèle.")
                
                st.markdown("### Cross-Validation")
                fig_cv = plot_cross_validation(results['cv_scores'])
                st.plotly_chart(fig_cv, use_container_width=True)
                
                st.markdown("### Rapport de Classification")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
                
                # Visualisation de quelques arbres
                st.markdown("### Visualisation de quelques Arbres de la Forêt")
                nb_to_display = min(2, n_estimators)
                for i in range(nb_to_display):
                    st.markdown(f"#### Arbre n°{i+1}")
                    fig, ax = plt.subplots(figsize=(12, 8))
                    tree.plot_tree(
                        model.estimators_[i],
                        filled=True,
                        feature_names=dataset.features_columns,
                        class_names=[str(cls) for cls in y.unique()],
                        ax=ax
                    )
                    st.pyplot(fig)
                
                # Cacher les paramètres après l'entraînement
                st.session_state.forest_trained = True
