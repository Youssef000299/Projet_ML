# 🍷 Application d'Analyse des Vins

Une application web interactive pour l'analyse, le nettoyage et la prédiction de la qualité des vins, développée avec Streamlit et scikit-learn.

## 📋 Table des matières

- [Présentation](#présentation)
- [Fonctionnalités](#fonctionnalités)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Flux de travail](#flux-de-travail)
- [Technologies utilisées](#technologies-utilisées)
- [Contribuer](#contribuer)

## 🎯 Présentation

Cette application permet d'analyser un jeu de données de vins, de nettoyer les données, d'entraîner différents modèles de machine learning et de comparer leurs performances. Elle offre une interface utilisateur intuitive et des visualisations interactives pour faciliter l'exploration et l'analyse des données.

## ✨ Fonctionnalités

- **Chargement et aperçu des données** : Importation et visualisation du jeu de données des vins
- **Exploration interactive** : Histogrammes, boxplots, matrices de corrélation et graphiques 3D
- **Nettoyage des données** : Gestion des valeurs manquantes, des outliers et normalisation
- **Entraînement de modèles** : Régression logistique, arbres de décision et random forest
- **Visualisation des modèles** : Matrices de confusion, importance des features, arbres de décision
- **Comparaison des performances** : Métriques d'évaluation et recommandation du meilleur modèle

## 🏗️ Structure du projet
Projet-ML/
├── app/
│ ├── dataset/ # Gestion des données
│ │ ├── forms/ # Formulaires pour la configuration des données
│ │ ├── models.py # Modèles de données
│ │ ├── services.py # Services pour le traitement des données
│ │ └── state.py # Gestion de l'état global des données
│ ├── layouts/ # Composants d'interface utilisateur
│ │ └── sidebar_components.py
│ ├── pages/ # Pages de l'application
│ │ ├── 0_Home.py # Page d'accueil
│ │ ├── 1_Dataset.py # Aperçu du dataset
│ │ ├── 2_Exploration_donnees.py # Exploration des données
│ │ ├── 3_Nettoyage_donnes.py # Nettoyage des données
│ │ ├── 4_Entrainement.py # Entraînement des modèles
│ │ └── 5_Comparaisons.py # Comparaison des modèles
│ ├── utils/ # Utilitaires
│ │ ├── model_storage.py # Stockage des modèles entraînés
│ │ └── plotly.py # Fonctions pour les visualisations Plotly
│ ├── main.py # Point d'entrée de l'application
│ ├── main_layout.py # Mise en page principale
│ └── routes.py # Configuration des routes
├── data/ # Données
│ └── vin.csv # Jeu de données des vins
└── requirements.txt # Dépendances Python


## 🚀 Installation

1. Clonez ce dépôt :
   git clone https://github.com/Youssef000299/Projet_ML.git
   
   cd Projet-ML

2. Créez un environnement virtuel et avtivez le :

   python -m venv .env

   Sur Windows : .env\Scripts\activate

3. Installez les dépendances :

   pip install -r requirements.txt

## 🖥️ Utilisation

1. Lancez l'application :
   ```bash
   cd Projet-ML
   streamlit run app/main.py
   ```

2. Ouvrez votre navigateur à l'adresse indiquée (généralement http://localhost:8501)

## 📊 Flux de travail

L'application suit un flux de travail séquentiel pour l'analyse des données :

1. **Aperçu du dataset** : Visualisation des données et sélection des variables
2. **Exploration des données** : Analyse approfondie des distributions et relations
3. **Nettoyage des données** : Traitement des valeurs manquantes et aberrantes
4. **Entraînement des modèles** : Configuration et entraînement de différents algorithmes
5. **Comparaison des modèles** : Évaluation des performances et sélection du meilleur modèle

## 🛠️ Technologies utilisées

- **Streamlit** : Framework pour l'interface utilisateur
- **Pandas** : Manipulation et analyse des données
- **NumPy** : Calculs numériques
- **Scikit-learn** : Algorithmes de machine learning
- **Plotly** : Visualisations interactives
- **Matplotlib** : Visualisations statiques

