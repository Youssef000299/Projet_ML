import streamlit as st
import pandas as pd
import main_layout
from routes import routes
from dataset.state import DatasetState

# Chargement du dataset vin.csv
if DatasetState.get_dataset() is None:
    data = pd.read_csv("data/vin.csv")
    
    class DatasetConfig:
        def __init__(self, data):
            self.filename = "vin.csv"
            self.data = data
            # Définir les colonnes features et target spécifiques au dataset vin
            self.features_columns = [col for col in data.columns if col != 'target']
            self.target_columns = ['target']
            self.data_raw = data
    
    dataset = DatasetConfig(data)
    DatasetState.set_dataset(dataset)

# Configuration de la page
st.set_page_config(page_title="Analyse des Vins", page_icon="🍷", layout="wide")

# Rendu de l'application
main_layout.sidebar()
pg = st.navigation(routes)
pg.run()

