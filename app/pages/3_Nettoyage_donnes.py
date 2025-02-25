import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import plotly.express as px

from dataset.state import DatasetState
from dataset.forms.config import dataset_config_form

# ---------------------------------------------------------------------
# Style CSS personnalisé pour un look moderne et élégant
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
            background: #f1f1f1;
            border-left: 4px solid #2575fc;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 4px;
            font-family: 'Arial', sans-serif;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header"><h1>🧹 Nettoyage des Données</h1></div>', unsafe_allow_html=True)

# Récupération du dataset
dataset = DatasetState.get_dataset()
df = dataset.data.copy()  # On travaille sur une copie initiale

# Création des onglets pour le nettoyage
tab_visual, tab_missing, tab_outliers, tab_norm = st.tabs([
    "Visualisation",
    "Gestion des valeurs manquantes",
    "Gestion des outliers",
    "Normalisation"
])

# ---------------------------------------------------------------------
# Onglet 1 : Visualisation des valeurs manquantes et aberrantes
# ---------------------------------------------------------------------
with tab_visual:
    st.subheader("Visualisation des valeurs manquantes et aberrantes")
    st.markdown("Sélectionnez les variables à analyser.")
    vars_visual = st.multiselect(
        "Variables",
        options=dataset.features_columns,
        default=dataset.features_columns
    )
    
    if vars_visual:
        # Graphique des valeurs manquantes
        missing_pct = df[vars_visual].isnull().mean() * 100
        st.markdown("**Pourcentage de valeurs manquantes**")
        st.bar_chart(missing_pct)
        
        # Détection des aberrants par la méthode IQR
        def detect_aberrants(series):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return series[(series < lower) | (series > upper)]
        
        # Calcul du nombre d'aberrants pour chaque variable sélectionnée
        outlier_counts = {}
        for var in vars_visual:
            aberrants = detect_aberrants(df[var].dropna())
            outlier_counts[var] = len(aberrants)
        
        # Transformation en DataFrame pour plotly
        outlier_df = pd.DataFrame({
            "Variable": list(outlier_counts.keys()),
            "Nb_Aberrants": list(outlier_counts.values())
        })
        
        st.markdown("**Nombre de valeurs aberrantes (méthode IQR)**")
        # Bar chart des outliers
        fig_outliers = px.bar(
            outlier_df,
            x="Variable",
            y="Nb_Aberrants",
            title="Valeurs aberrantes détectées (IQR)",
            labels={"Variable": "Variables", "Nb_Aberrants": "Nombre d'Aberrants"},
            color="Variable"
        )
        fig_outliers.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.05)',
            font=dict(family="Arial, sans-serif")
        )
        st.plotly_chart(fig_outliers, use_container_width=True)
    else:
        st.info("Sélectionnez au moins une variable.")

# ---------------------------------------------------------------------
# Onglet 2 : Gestion des valeurs manquantes
# ---------------------------------------------------------------------
with tab_missing:
    st.subheader("Gestion des valeurs manquantes")
    st.markdown("""
    **Concepts :**
    - **Suppression** : Retirer les lignes contenant des valeurs manquantes.
    - **Imputation** : Remplacer les valeurs manquantes par une estimation (moyenne pour les numériques, mode pour les catégoriques).
    """)
    missing_method = st.radio(
        "Choisissez la méthode de traitement",
        options=["Supprimer les échantillons", "Imputer les valeurs"]
    )
    vars_missing = st.multiselect(
        "Variables concernées",
        options=dataset.features_columns,
        default=dataset.features_columns
    )
    
    if st.button("Appliquer le traitement des valeurs manquantes"):
        df_missing = df.copy()
        if missing_method == "Supprimer les échantillons":
            df_missing = df_missing.dropna(subset=vars_missing)
        else:
            for var in vars_missing:
                if pd.api.types.is_numeric_dtype(df_missing[var]):
                    # Imputation par la moyenne pour les numériques
                    df_missing[var].fillna(df_missing[var].mean(), inplace=True)
                else:
                    # Imputation par le mode pour les catégoriques
                    df_missing[var].fillna(df_missing[var].mode()[0], inplace=True)
        
        st.success("Traitement des valeurs manquantes effectué.")
        st.markdown("**Aperçu des données après traitement**")
        st.dataframe(df_missing.head())
        
        st.markdown("**Comparaison des taux de valeurs manquantes**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Avant traitement")
            st.bar_chart(df[vars_missing].isnull().mean() * 100)
        with col2:
            st.markdown("Après traitement")
            st.bar_chart(df_missing[vars_missing].isnull().mean() * 100)
        
        # Enregistrer les nouvelles données directement dans dataset.data
        dataset.data = df_missing
        df = dataset.data  # Mettre à jour df pour les prochains onglets

# ---------------------------------------------------------------------
# Onglet 3 : Gestion des outliers
# ---------------------------------------------------------------------
with tab_outliers:
    st.subheader("Gestion des outliers")
    st.markdown("""
    **Concepts :**
    - **Méthode IQR** : Un point est aberrant s'il est en dehors de [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    - **Méthode Z-score** : Un point est aberrant si son Z-score est supérieur à 3.
    """)
    outlier_method = st.radio(
        "Choisissez la méthode de traitement des outliers",
        options=["Méthode IQR", "Méthode Z-score"]
    )
    var_outlier = st.selectbox(
        "Sélectionnez une variable",
        options=dataset.features_columns
    )
    
    if st.button("Traiter les outliers"):
        df_outliers = df.copy()
        if outlier_method == "Méthode IQR":
            Q1 = df_outliers[var_outlier].quantile(0.25)
            Q3 = df_outliers[var_outlier].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            median_val = df_outliers[var_outlier].median()
            # Remplace les outliers par la médiane
            df_outliers[var_outlier] = np.where(
                (df_outliers[var_outlier] < lower_bound) | (df_outliers[var_outlier] > upper_bound),
                median_val,
                df_outliers[var_outlier]
            )
        else:
            z_scores = np.abs((df_outliers[var_outlier] - df_outliers[var_outlier].mean()) / df_outliers[var_outlier].std())
            median_val = df_outliers[var_outlier].median()
            df_outliers[var_outlier] = np.where(z_scores > 3, median_val, df_outliers[var_outlier])
        
        st.success("Traitement des outliers appliqué.")
        st.markdown("**Aperçu des données après traitement**")
        st.dataframe(df_outliers.head())
        
        st.markdown("**Comparaison avant/après traitement**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Avant traitement")
            st.line_chart(df[var_outlier])
        with col2:
            st.markdown("Après traitement")
            st.line_chart(df_outliers[var_outlier])
        
        # Enregistrer les nouvelles données directement dans dataset.data
        dataset.data = df_outliers
        df = dataset.data

# ---------------------------------------------------------------------
# Onglet 4 : Normalisation
# ---------------------------------------------------------------------
with tab_norm:
    st.subheader("Normalisation des Données")
    st.markdown("""
    **Concepts :**
    - **Standard Scaler** : Standardise les données pour avoir une moyenne de 0 et un écart-type de 1.
    - **Min-Max Scaler** : Ramène les données dans l'intervalle [0, 1].
    - **Robust Scaler** : Utilise la médiane et l'IQR pour atténuer l'effet des outliers.
    """)
    norm_method = st.selectbox(
        "Méthode de normalisation",
        options=["Standard Scaler", "Min-Max Scaler", "Robust Scaler"]
    )
    norm_vars = st.multiselect(
        "Variables à normaliser",
        options=dataset.features_columns,
        default=dataset.features_columns
    )
    
    if st.button("Appliquer la normalisation"):
        df_norm = df.copy()
        if norm_method == "Standard Scaler":
            scaler = StandardScaler()
        elif norm_method == "Min-Max Scaler":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
        
        df_norm[norm_vars] = scaler.fit_transform(df_norm[norm_vars])
        
        st.success("Normalisation appliquée.")
        st.markdown("**Aperçu des données normalisées**")
        st.dataframe(df_norm.head())
        
        st.markdown("**Comparaison des statistiques avant/après normalisation**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Avant")
            st.dataframe(df[norm_vars].describe().round(2))
        with col2:
            st.markdown("Après")
            st.dataframe(df_norm[norm_vars].describe().round(2))
        
        # Enregistrer les nouvelles données directement dans dataset.data
        dataset.data = df_norm
        df = dataset.data
