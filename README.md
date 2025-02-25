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
![image](https://github.com/user-attachments/assets/3a368547-60d5-46cd-8778-c43d2e31c8df)



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

