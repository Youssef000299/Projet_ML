import streamlit as st
import pandas as pd
from dataset.state import DatasetState

# Style cohérent avec les autres pages
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
        
        .section-header {
            background: linear-gradient(45deg, var(--primary-color) 30%, var(--secondary-color));
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .data-container {
            background-color: var(--feature-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 2rem 0;
        }
        
        .column-section {
            background-color: var(--info-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            height: 100%;
        }
        
        /* Variables pour le mode clair/sombre */
        [data-testid="stAppViewContainer"] {
            --primary-color: #722F37;
            --secondary-color: #9A3324;
            --feature-bg: rgba(255, 255, 255, 0.05);
            --info-bg: rgba(255, 255, 255, 0.05);
            --border-color: rgba(255, 255, 255, 0.1);
        }
        
        /* Style pour les tableaux */
        [data-testid="stDataFrame"] {
            background-color: var(--feature-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1rem;
            margin: 1.5rem 0;
        }
        
        /* Style pour le sélecteur de colonnes */
        .column-selector {
            background-color: var(--feature-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 2rem 0;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header"><h1>📊 Aperçu du Dataset des Vins</h1></div>', unsafe_allow_html=True)

# Récupération du dataset
dataset = DatasetState.get_dataset()
df = dataset.data

# Initialiser la sélection dans st.session_state si elle n'existe pas
default_features = [col for col in df.columns if col != 'target']
if "selected_features" not in st.session_state:
    st.session_state["selected_features"] = default_features

# Sélection des colonnes
with st.container():
    st.markdown("""
        <div class="column-selector">
            <h3>🎯 Sélection des Variables</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # On n'utilise pas key="selected_features" pour éviter le conflit
        features = st.multiselect(
            "Variables prédictives",
            options=[col for col in df.columns if col != 'target'],
            default=st.session_state["selected_features"],  # Utilise la valeur stockée dans session_state
            help="Sélectionnez les variables qui serviront à prédire la cible"
        )
    
    with col2:
        target = st.selectbox(
            "Variable cible",
            options=['target'],
            disabled=True,
            help="Variable à prédire (qualité du vin)"
        )
    
    # Bouton pour enregistrer la sélection
    if st.button("Enregistrer la sélection"):
        # On met à jour la sélection dans st.session_state et dataset
        st.session_state["selected_features"] = features
        dataset.features_columns = features
        dataset.target_columns = [target]
        st.success("✅ Sélection des variables enregistrée avec succès!")

# Visualisation des variables sélectionnées
with st.container():
    st.markdown("""
        <div class="data-container">
            <h3>📊 Aperçu des Variables Sélectionnées</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
            <div class="column-section">
                <h4>Variables explicatives</h4>
                <p>Variables utilisées pour faire les prédictions</p>
            </div>
        """, unsafe_allow_html=True)
        
        # On récupère la liste finale de features depuis st.session_state
        selected_features = st.session_state["selected_features"]
        features_df = df[selected_features].head()
        st.dataframe(features_df, use_container_width=True)
        
        with st.expander("📊 Statistiques des features"):
            st.dataframe(
                df[selected_features].describe().round(2),
                use_container_width=True
            )
    
    with col2:
        st.markdown("""
            <div class="column-section">
                <h4>Variable cible</h4>
                <p>Variable à prédire</p>
            </div>
        """, unsafe_allow_html=True)
        
        target_df = df[['target']].head()
        st.dataframe(target_df, use_container_width=True)
        
        with st.expander("📊 Distribution de la target"):
            target_dist = df['target'].value_counts()
            st.bar_chart(target_dist)

# Aperçu complet des données
with st.container():
    st.markdown("""
        <div class="data-container">
            <h3>🔍 Aperçu complet des données</h3>
        </div>
    """, unsafe_allow_html=True)
    
    n_rows = st.slider("Nombre de lignes à afficher", 5, len(df), 10)
    st.dataframe(df.head(n_rows), use_container_width=True)
