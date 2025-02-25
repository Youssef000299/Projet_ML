import streamlit as st

class ModelStorage:
    """Classe pour stocker les résultats des modèles entraînés"""
    
    @staticmethod
    def store_model_results(model_name, results):
        """
        Stocke les résultats d'un modèle dans la session Streamlit
        
        Args:
            model_name (str): Nom du modèle (ex: "Logistic Regression")
            results (dict): Dictionnaire contenant les résultats du modèle
        """
        if "trained_models" not in st.session_state:
            st.session_state.trained_models = {}
        
        st.session_state.trained_models[model_name] = results
    
    @staticmethod
    def get_model_results(model_name=None):
        """
        Récupère les résultats d'un modèle ou de tous les modèles
        
        Args:
            model_name (str, optional): Nom du modèle à récupérer. Si None, retourne tous les modèles.
        
        Returns:
            dict: Résultats du modèle ou dictionnaire de tous les modèles
        """
        if "trained_models" not in st.session_state:
            st.session_state.trained_models = {}
        
        if model_name:
            return st.session_state.trained_models.get(model_name, None)
        else:
            return st.session_state.trained_models
    
    @staticmethod
    def has_trained_models():
        """
        Vérifie si des modèles ont été entraînés
        
        Returns:
            bool: True si des modèles ont été entraînés, False sinon
        """
        return "trained_models" in st.session_state and len(st.session_state.trained_models) > 0 