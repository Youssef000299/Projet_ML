import streamlit as st
from dataset.state import DatasetState
from layouts.sidebar_components import render_dataset_stats

# Sidebar de la page d'accueil
dataset = DatasetState.get_dataset()

# Style adaptatif pour le mode clair/sombre
st.markdown("""
    <style>
        /* Style adaptatif */
        [data-testid="stAppViewContainer"] {
            background: var(--background-color);
        }
        
        .main-header {
            background: linear-gradient(45deg, var(--primary-color) 30%, var(--secondary-color));
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .feature-container {
            background-color: var(--feature-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            transition: transform 0.2s;
        }
        
        .feature-container:hover {
            transform: translateY(-5px);
        }
        
        .info-box {
            background-color: var(--info-bg);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            border: 1px solid var(--border-color);
        }
        
        /* Variables pour le mode clair */
        [data-testid="stAppViewContainer"] {
            --primary-color: #722F37;
            --secondary-color: #9A3324;
            --feature-bg: rgba(255, 255, 255, 0.05);
            --info-bg: rgba(255, 255, 255, 0.05);
            --border-color: rgba(255, 255, 255, 0.1);
            --text-color: inherit;
        }
        
        /* Ajustements pour les conteneurs */
        .stMetric {
            background-color: var(--feature-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
            padding: 1rem !important;
        }
        
        /* Style pour l'image */
        .wine-image-container {
            display: flex;
            justify-content: center;
            margin: 2rem 0;
        }
        .wine-image-container img {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 100%;
            height: auto;
        }
    </style>
    """, unsafe_allow_html=True)

# En-tête principal avec image
st.markdown("""
    <div class="main-header">
        <h1>🍷 Analyse des Vins</h1>
        <p style="font-size: 1.2em; margin-top: 1rem;">
            Une plateforme d'analyse de données et d'apprentissage automatique dédiée aux vins
        </p>
    </div>
    """, unsafe_allow_html=True)

# Image centrale
st.markdown("""
    <div class="wine-image-container">
        <img src="https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?auto=format&fit=crop&w=800" 
             alt="Verres de vin rouge dans une cave"
             style="max-width: 800px;">
    </div>
    """, unsafe_allow_html=True)

# Introduction
st.markdown("""
    <div class="info-box">
        Cette application vous permet d'explorer et d'analyser un jeu de données sur les vins, 
        en utilisant différentes techniques de visualisation et d'apprentissage automatique.
    </div>
    """, unsafe_allow_html=True)

# Fonctionnalités principales
st.markdown("### 🎯 Fonctionnalités principales")

# Utilisation de containers pour un meilleur espacement
col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.markdown("""
            <div class="feature-container">
                <h4>📊 Explorer et Comprendre</h4>
                <p>Visualisez et analysez les données à travers des graphiques interactifs</p>
            </div>
            """, unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown("""
            <div class="feature-container">
                <h4>🤖 Entraîner des Modèles</h4>
                <p>Testez différents algorithmes d'apprentissage automatique</p>
            </div>
            """, unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown("""
            <div class="feature-container">
                <h4>📈 Évaluer les Performances</h4>
                <p>Analysez et comparez les résultats de vos modèles</p>
            </div>
            """, unsafe_allow_html=True)

# À propos du dataset
st.markdown("""
    <div class="info-box">
        <h3>💡 À propos du dataset</h3>
        <p>Notre jeu de données contient diverses caractéristiques des vins, permettant de :</p>
        <ul>
            <li>🔍 Analyser les propriétés chimiques des vins</li>
            <li>🔗 Comprendre les relations entre ces propriétés</li>
            <li>⭐ Prédire la qualité des vins</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Message de démarrage dans un container
with st.container():
    st.success("""
        #### ✨ Le jeu de données des vins est chargé et prêt à être analysé !

        Rendez-vous dans la section **"Exploration des données"** pour commencer votre analyse !
        """)

# Métriques du dataset dans un container
with st.container():
    st.markdown("### 📊 Statistiques du Dataset")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Échantillons",
            f"{len(dataset.data):,}",
            help="Nombre total d'échantillons de vin dans le dataset"
        )

    with col2:
        st.metric(
            "Variables",
            len(dataset.features_columns),
            help="Caractéristiques mesurées pour chaque vin"
        )

    with col3:
        st.metric(
            "Classes",
            len(dataset.data[dataset.target_columns[0]].unique()),
            help="Nombre de niveaux de qualité différents"
        )
