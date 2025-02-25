from streamlit import Page

routes = [
    Page("pages/0_Home.py", title="Accueil", icon="🏠"),
    Page("pages/1_Dataset.py", title="Jeu de données", icon="📊"),
    Page("pages/2_Exploration_donnees.py", title="Exploration", icon="🔍"),
    Page("pages/3_Nettoyage_donnes.py", title="Nettoyage", icon="🧹"),
    Page("pages/4_Entrainement.py", title="Entraînement", icon="🤖"),
    Page("pages/5_Comparaisons.py", title="Comparaisons", icon="📈"),
]