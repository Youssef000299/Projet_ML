import streamlit as st
from dataset.state import DatasetState

def render_dataset_manager():
    """Affiche les informations du dataset des vins."""
    dataset = DatasetState.get_dataset()
    
    st.info(f"""
    ### Dataset des Vins
    
    📊 **{dataset.filename}**  
    📈 {len(dataset.data.columns)} colonnes | {len(dataset.data)} lignes
    """)

def render_dataset_stats():
    """Affiche les statistiques du dataset dans la sidebar."""
    dataset = DatasetState.get_dataset()
    st.markdown("### Statistiques")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Variables",
            len(dataset.features_columns),
            "features",
            delta_color="normal",
            help="Nombre de variables explicatives",
        )
    with col2:
        st.metric(
            "Cibles",
            len(dataset.target_columns),
            "targets",
            delta_color="normal",
            help="Nombre de variables à prédire",
        )
