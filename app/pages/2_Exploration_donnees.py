import math
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import linregress  # Pour le calcul du R²

from dataset.state import DatasetState
from dataset.forms.config import dataset_config_form
from utils.plotly import get_color_palette

# ---------------------------------------------------------------------
# Style cohérent
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
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header"><h1>🔍 Exploration des Données</h1></div>', unsafe_allow_html=True)

# ====================
#  Fonctions de visualisation
# ====================

def create_histogram_plot(data, features, target=None):
    n_cols = 3
    n_rows = math.ceil(len(features) / n_cols)
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=features,
        horizontal_spacing=0.1,
        vertical_spacing=0.1
    )
    
    for i, feature in enumerate(features):
        row = i // n_cols + 1
        col = i % n_cols + 1
        if target and target in data.columns:
            unique_classes = data[target].unique()
            colors = get_color_palette(len(unique_classes))
            color_map = dict(zip(unique_classes, colors))
            for cls in unique_classes:
                subset = data[data[target] == cls][feature].dropna()
                if len(subset) < 2:
                    continue
                fig.add_trace(
                    go.Histogram(
                        x=subset,
                        name=str(cls),
                        opacity=0.7,
                        marker=dict(color=color_map[cls]),
                        showlegend=(i == 0)
                    ),
                    row=row,
                    col=col
                )
        else:
            x_data = data[feature].dropna()
            if len(x_data) < 2:
                continue
            fig.add_trace(
                go.Histogram(
                    x=x_data,
                    showlegend=False,
                    marker=dict(color='#1f77b4')
                ),
                row=row,
                col=col
            )
        fig.update_xaxes(title_text="Valeurs", row=row, col=col)
        fig.update_yaxes(title_text="Fréquence", row=row, col=col)
    
    fig.update_layout(
        height=300 * n_rows,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.05)',
        font=dict(family="Arial, sans-serif", size=12),
        barmode='overlay',
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1
        )
    )
    return fig

def create_3d_scatter_plot(data, x, y, z, color=None):
    fig = px.scatter_3d(
        data,
        x=x,
        y=y,
        z=z,
        color=color,
        title=f"Nuage de points 3D: {x} vs {y} vs {z}"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.05)',
        font=dict(family="Arial, sans-serif")
    )
    return fig

def create_boxplot(data, features, target=None):
    fig = go.Figure()
    if target:
        for feature in features:
            for target_value in data[target].unique():
                subset = data[data[target] == target_value][feature]
                fig.add_trace(go.Box(
                    y=subset,
                    name=str(target_value),
                    legendgroup=feature,
                    legendgrouptitle_text=feature,
                    boxpoints='outliers',
                    pointpos=0,
                    jitter=0
                ))
        fig.update_layout(
            boxmode='group',
            title_text="Boxplots groupés par classe",
            yaxis_title="Valeur",
            xaxis_title="Classe",
            height=max(400, 100 * len(features)),
            showlegend=True,
            legend=dict(
                groupclick="toggleitem",
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1
            )
        )
    else:
        for feature in features:
            fig.add_trace(go.Box(
                y=data[feature],
                name=feature,
                boxpoints='outliers',
                jitter=0,
                pointpos=0
            ))
        fig.update_layout(
            title_text="Boxplots des variables",
            yaxis_title="Valeur",
            xaxis_title="Variable",
            showlegend=False,
            height=600
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.05)',
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>" +
                      "Max: %{upperbound}<br>" +
                      "Q3: %{q3}<br>" +
                      "Médiane: %{median}<br>" +
                      "Q1: %{q1}<br>" +
                      "Min: %{lowerbound}<br>" +
                      "<extra></extra>"
    )
    return fig

def create_correlation_heatmap(data, features):
    corr_matrix = data[features].corr()
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    corr_matrix_masked = corr_matrix.copy()
    corr_matrix_masked[mask] = np.nan
    heat_map_text = np.round(corr_matrix_masked, 2)
    heat_map_text[mask] = ""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix_masked,
        x=features,
        y=features,
        text=heat_map_text,
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1
    ))
    fig.update_layout(
        title='Matrice de Corrélation',
        height=600,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.05)'
    )
    return fig

def create_violin_plot(data, features, target=None):
    if target and target in data.columns:
        df_long = data.melt(id_vars=[target], value_vars=features, var_name="Variable", value_name="Value")
        fig = px.violin(df_long, x="Variable", y="Value", color=target, box=True, points="all",
                        title="Violin Plot par variable et par classe")
    else:
        df_long = data[features].melt(var_name="Variable", value_name="Value")
        fig = px.violin(df_long, x="Variable", y="Value", box=True, points="all",
                        title="Violin Plot (toutes variables)", color="Variable")
        fig.update_layout(showlegend=False)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.05)',
        font=dict(family="Arial, sans-serif")
    )
    return fig

def create_missing_values_plot(data):
    missing_percent = data.isnull().mean() * 100
    missing_df = pd.DataFrame({"Variable": missing_percent.index, "Taux (%)": missing_percent.values})
    fig = px.bar(
        missing_df,
        x="Variable",
        y="Taux (%)",
        title="Pourcentage de valeurs manquantes",
        text="Taux (%)",
        color="Variable"
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.05)',
        xaxis_tickangle=-45,
        font=dict(family="Arial, sans-serif")
    )
    return fig

def create_scatter_regression_plot(data, x, y):
    df_temp = data[[x, y]].dropna()
    if x == y:
        fig = go.Figure()
        fig.add_annotation(
            text="Impossible de tracer une régression si X et Y sont identiques !",
            showarrow=False,
            font=dict(color='red', size=16)
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.05)',
            title="Erreur : même variable en X et Y"
        )
        return fig
    if len(df_temp) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Pas assez de données pour tracer une régression linéaire.",
            showarrow=False,
            font=dict(color='orange', size=16)
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.05)',
            title="Données insuffisantes"
        )
        return fig
    slope, intercept = np.polyfit(df_temp[x], df_temp[y], 1)
    regression_line = slope * df_temp[x] + intercept
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_temp[x],
        y=df_temp[y],
        mode='markers',
        name='Données',
        marker=dict(color='#1f77b4')
    ))
    fig.add_trace(go.Scatter(
        x=df_temp[x],
        y=regression_line,
        mode='lines',
        name='Régression',
        line=dict(color='red', width=2)
    ))
    fig.update_layout(
        title=f"Régression Linéaire: {x} vs {y}",
        xaxis_title=x,
        yaxis_title=y,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.05)',
        font=dict(family="Arial, sans-serif")
    )
    return fig

# ---------------------------------------------------------------------
# Conversion forcée si la cible n'est pas numérique
# ---------------------------------------------------------------------
def create_feature_target_correlation_plot(data, features, target):
    """
    Crée un bar chart de la corrélation de chaque feature (numérique) avec la variable cible.
    Si la cible n'est pas numérique (par ex. classes), on la convertit en codes.
    """
    data_corr = data.copy()
    
    # Si la cible est catégorielle, on la convertit en codes entiers
    if not pd.api.types.is_numeric_dtype(data_corr[target]):
        data_corr[target] = data_corr[target].astype("category").cat.codes
    
    corr_values = []
    for feat in features:
        # On calcule la corrélation seulement si la feature est numérique
        if pd.api.types.is_numeric_dtype(data_corr[feat]):
            c = data_corr[feat].corr(data_corr[target])
            corr_values.append((feat, c))
    
    if not corr_values:
        return None  # Pas de feature numérique
    
    df_corr = pd.DataFrame(corr_values, columns=["Feature", "Correlation"])
    fig = px.bar(
        df_corr,
        x="Feature",
        y="Correlation",
        title="Corrélation des Features avec la Variable Cible",
        labels={"Feature": "Caractéristiques", "Correlation": "Corrélation"},
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.05)',
        font=dict(family="Arial, sans-serif")
    )
    return fig

# =================================================================
# PAGE PRINCIPALE
# =================================================================

def page():
    dataset = DatasetState.get_dataset()
    df = dataset.data

    st.markdown("# Explorations & Visualisations")
    st.markdown("""
    **Bienvenue dans l'outil d'exploration !**  
    Vous trouverez ci-dessous différentes analyses pour comprendre vos données :
    - Statistiques descriptives  
    - Histogrammes univariés  
    - Boxplots  
    - Corrélations  
    - Matrice de dispersion  
    - Graphiques avancés (violin plot, valeurs manquantes, régression...)  
    """)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Nb. de lignes", len(df))
    with c2:
        st.metric("Nb. de features", len(dataset.features_columns))
    with c3:
        st.metric("Nb. de targets", len(dataset.target_columns))
    with c4:
        if dataset.target_columns:
            st.metric("Classes (target)", len(df[dataset.target_columns[0]].unique()))

    tab_overview, tab_hist, tab_box, tab_corr, tab_pair, tab_extra, tab_3d = st.tabs([
        "Aperçu & Stats",
        "Histogrammes Univariés",
        "Boxplots",
        "Corrélations",
        "Matrice de Dispersion",
        "Analyses Avancées",
        "Graphiques 3D"
    ])

    # --------------------------------------------------------------------
    # Onglet : Aperçu & Stats
    # --------------------------------------------------------------------
    with tab_overview:
        st.subheader("Aperçu des données")
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Types et valeurs manquantes**")
            info_df = pd.DataFrame({
                "Type": df.dtypes,
                "Non-nulls": df.count(),
                "Manquantes (%)": (df.isna().sum() / len(df) * 100).round(2)
            })
            st.dataframe(info_df, use_container_width=True)
        with colB:
            st.markdown("**Statistiques descriptives**")
            st.dataframe(df.describe().round(2), use_container_width=True)
        st.markdown("**Extrait du DataFrame**")
        st.dataframe(df.head(10), use_container_width=True)

    # --------------------------------------------------------------------
    # Onglet : Histogrammes Univariés
    # --------------------------------------------------------------------
    with tab_hist:
        st.subheader("Histogrammes Univariés")
        selected_features = st.multiselect(
            "Choisissez les variables à explorer",
            options=dataset.features_columns,
            default=dataset.features_columns[:4]
        )
        if selected_features:
            fig_hist = create_histogram_plot(df, selected_features, dataset.target_columns[0] if dataset.target_columns else None)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Sélectionnez au moins une variable.")

    # --------------------------------------------------------------------
    # Onglet : Boxplots
    # --------------------------------------------------------------------
    with tab_box:
        st.subheader("Boxplots et dispersion")
        box_features = st.multiselect(
            "Variables pour les boxplots",
            options=dataset.features_columns,
            default=dataset.features_columns[:4]
        )
        if box_features:
            group_by_target = False
            target = None
            if dataset.target_columns:
                group_by_target = st.checkbox("Séparer par la cible", value=False)
                if group_by_target:
                    target = dataset.target_columns[0]
            fig_box = create_boxplot(df, box_features, target)
            st.plotly_chart(fig_box, use_container_width=True)
            if st.checkbox("Afficher des stats supplémentaires"):
                stats_df = df[box_features].describe()
                stats_df.loc['skewness'] = df[box_features].skew()
                stats_df.loc['kurtosis'] = df[box_features].kurtosis()
                st.dataframe(stats_df.style.format("{:.2f}").background_gradient(cmap='RdYlBu'), use_container_width=True)
        else:
            st.info("Veuillez sélectionner au moins une variable pour les boxplots.")

    # --------------------------------------------------------------------
    # Onglet : Corrélations
    # --------------------------------------------------------------------
    with tab_corr:
        st.subheader("Matrice de corrélation")
        corr_features = st.multiselect(
            "Variables à corréler",
            options=dataset.features_columns,
            default=dataset.features_columns[:5]
        )
        if len(corr_features) > 1:
            fig_corr = create_correlation_heatmap(df, corr_features)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Sélectionnez au moins deux variables.")

        # Corrélation entre chaque feature et la cible (même si la cible est catégorielle)
        if dataset.target_columns:
            target_col = dataset.target_columns[0]
            st.markdown("### Corrélation des Features avec la Variable Cible")
            corr_fig = create_feature_target_correlation_plot(df, dataset.features_columns, target_col)
            if corr_fig is not None:
                st.plotly_chart(corr_fig, use_container_width=True)
            else:
                st.info("Aucune feature numérique pour calculer la corrélation ou cible introuvable.")

    # --------------------------------------------------------------------
    # Onglet : Matrice de Dispersion
    # --------------------------------------------------------------------
    with tab_pair:
        st.subheader("Scatter Matrix (Pairplot)")
        pair_features = st.multiselect(
            "Sélectionnez jusqu'à 6 variables",
            options=dataset.features_columns,
            default=dataset.features_columns[:4]
        )
        if len(pair_features) > 1:
            fig_pair = px.scatter_matrix(
                df,
                dimensions=pair_features,
                color=dataset.target_columns[0] if dataset.target_columns else None,
                title="Matrice de dispersion",
                labels={col: col.replace('_', ' ').title() for col in pair_features}
            )
            fig_pair.update_traces(
                diagonal_visible=False,
                showupperhalf=False,
                hovertemplate='%{xaxis.title.text}: %{x}<br>%{yaxis.title.text}: %{y}<br>'
            )
            fig_pair.update_layout(
                height=700,
                width=700,
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.05)',
                title_x=0.5,
            )
            st.plotly_chart(fig_pair, use_container_width=True)
            if len(pair_features) > 6:
                st.warning("Avec plus de 6 variables, le scatter matrix devient difficile à lire.")
        else:
            st.info("Sélectionnez au moins 2 variables pour la matrice de dispersion.")

    # --------------------------------------------------------------------
    # Onglet : Analyses Avancées
    # --------------------------------------------------------------------
    with tab_extra:
        st.subheader("Analyses Avancées")
        st.markdown("### Scatter Plot + Régression Linéaire")
        col_x, col_y = st.columns(2)
        default_y_index = 1 if len(dataset.features_columns) > 1 else 0
        with col_x:
            scatter_x = st.selectbox("Variable X", options=dataset.features_columns, index=0, key="scatter_x")
        with col_y:
            scatter_y = st.selectbox("Variable Y", options=dataset.features_columns, index=default_y_index, key="scatter_y")
        if scatter_x and scatter_y:
            fig_scatter = create_scatter_regression_plot(df, scatter_x, scatter_y)
            st.plotly_chart(fig_scatter, use_container_width=True)
            if scatter_x != scatter_y:
                temp_df = df[[scatter_x, scatter_y]].dropna()
                if len(temp_df) >= 2:
                    slope, intercept, r_value, p_value, std_err = linregress(temp_df[scatter_x], temp_df[scatter_y])
                    r_squared = r_value ** 2
                    st.markdown(f"**Coefficient de détermination R² : {r_squared:.3f}**")
                    if r_squared >= 0.7:
                        st.markdown("La régression linéaire est **plutôt bonne** (R² élevé).")
                    elif 0.4 <= r_squared < 0.7:
                        st.markdown("La régression linéaire est **moyenne** (R² modéré).")
                    else:
                        st.markdown("La régression linéaire est **faible** (R² bas).")
        else:
            st.info("Sélectionnez deux variables différentes pour la régression.")

    # --------------------------------------------------------------------
    # Onglet : Graphiques 3D
    # --------------------------------------------------------------------
    with tab_3d:
        st.subheader("Nuage de points 3D")
        st.markdown("Sélectionnez trois variables numériques pour générer un nuage de points 3D.")
        col_x3, col_y3, col_z3 = st.columns(3)
        with col_x3:
            var_x = st.selectbox("Variable X", options=dataset.features_columns, index=0, key="3d_x")
        with col_y3:
            var_y = st.selectbox("Variable Y", options=dataset.features_columns, index=1, key="3d_y")
        with col_z3:
            var_z = st.selectbox("Variable Z", options=dataset.features_columns, index=2 if len(dataset.features_columns)>2 else 0, key="3d_z")
        color_option = None
        if dataset.target_columns:
            if st.checkbox("Colorer par la cible", value=False, key="3d_color"):
                color_option = dataset.target_columns[0]
        if var_x and var_y and var_z:
            fig_3d = create_3d_scatter_plot(df, var_x, var_y, var_z, color_option)
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("Sélectionnez trois variables pour le graphique 3D.")

# =================================================================
# SQUELETTE SI PAS DE DATASET
# =================================================================

def render_no_dataset_skeleton():
    st.markdown("# Explorations & Visualisations")
    st.markdown("Aucun jeu de données n'est actuellement chargé. Veuillez en importer un.")
    with st.container():
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("### :warning: Dataset manquant")
            st.caption("Importez vos données pour pouvoir démarrer l'analyse.")
            st.button("Charger un Dataset", type="primary", on_click=dataset_config_form, use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Lignes", "---")
    with c2:
        st.metric("Features", "---")
    with c3:
        st.metric("Targets", "---")
    with c4:
        st.metric("Classes", "---")
    tabs = st.tabs(["Aperçu & Stats", "Histogrammes Univariés", "Boxplots", "Corrélations", "Matrice de Dispersion", "Analyses Avancées", "Graphiques 3D"])
    for t in tabs:
        with t:
            st.info("Veuillez charger un dataset pour voir le contenu.")

# =================================================================
# LANCEMENT DE LA PAGE
# =================================================================

dataset = DatasetState.get_dataset()
if not dataset:
    render_no_dataset_skeleton()
else:
    page()
