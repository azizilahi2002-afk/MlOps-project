# Importation de FastAPI pour créer l'API
from fastapi import FastAPI, HTTPException

# Importation de BaseModel de Pydantic pour la validation des données
from pydantic import BaseModel

# Importation de joblib pour charger le modèle
import joblib

# Importation de numpy pour les calculs
import numpy as np

# Importation de Path pour gérer les chemins de fichiers
from pathlib import Path

# Importation de logging pour journaliser les appels
import logging

# Importation de datetime pour timestamp
from datetime import datetime
import csv
PREDICTION_LOG_PATH = Path("data/predictions_log.csv")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Schéma pour les données d'entrée (ce que l'API reçoit)
class PredictionInput(BaseModel):
    ca_total: float      # Chiffre d'affaires total
    nb_commandes: int    # Nombre de commandes
    
    # Exemple de valeurs par défaut (optionnel)
    class Config:
        json_schema_extra = {
            "example": {
                "ca_total": 15000.0,
                "nb_commandes": 320
            }
        }

# Schéma pour les données de sortie (ce que l'API renvoie)
class PredictionOutput(BaseModel):
    prediction_ca_jour_suivant: float  # Prédiction du CA du jour suivant
    inputs: PredictionInput            # Données d'entrée pour traçabilité
    timestamp: str                     # Horodatage de la prédiction
    model_version: str                 # Version du modèle utilisé
# Création de l'application FastAPI
app = FastAPI(
    title="API de Prédiction de CA",
    description="API pour prédire le chiffre d'affaires du jour suivant",
    version="1.0.0"
)

# Variable globale pour stocker le modèle
model = None

# Fonction pour charger le modèle au démarrage
def load_model():
    """
    Charge le modèle entraîné depuis le fichier .pkl
    """
    global model
    try:
        # Chemin vers le modèle sauvegardé
        model_path = Path("C:/Users/azizi/mini-mlops-project/models/random_forest_model.pkl")
        
        logger.info(f"Chargement du modèle depuis : {model_path}")
        
        # Chargement du modèle
        model = joblib.load(model_path)
        
        logger.info("Modèle chargé avec succès")
        logger.info(f"Type de modèle : {type(model)}")
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {e}")
        raise e

# Événement de démarrage de l'API
@app.on_event("startup")
async def startup_event():
    """
    Événement déclenché au démarrage de l'API
    """
    logger.info("Démarrage de l'API...")
    load_model()
    logger.info("API prête à recevoir des requêtes")
@app.get("/")
async def root():
    """
    Endpoint racine - simple message de bienvenue
    """
    return {
        "message": "API de prédiction de chiffre d'affaires",
        "endpoints": {
            "GET /": "Cette page",
            "GET /health": "Vérifie l'état de l'API",
            "POST /predict": "Faire une prédiction",
            "GET /docs": "Documentation Swagger UI",
            "GET /redoc": "Documentation Redoc"
        }
    }

@app.get("/health")
async def health_check():
    """
    Endpoint de santé - vérifie que l'API est fonctionnelle
    """
    try:
        # Vérifie si le modèle est chargé
        if model is None:
            status = "MODEL_NOT_LOADED"
            message = "Le modèle n'est pas chargé"
        else:
            status = "HEALTHY"
            message = "API et modèle opérationnels"
        
        return {
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model is not None
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du health check: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")
@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Endpoint pour faire une prédiction du CA du jour suivant
    
    Args:
        input_data: Données d'entrée contenant ca_total et nb_commandes
    
    Returns:
        PredictionOutput: Prédiction du CA du jour suivant avec les inputs
    """
    try:
        # Journalisation de l'appel
        logger.info(f"Requête de prédiction reçue: {input_data}")
        
        # Vérifie si le modèle est chargé
        if model is None:
            logger.error("Modèle non chargé")
            raise HTTPException(
                status_code=503,
                detail="Modèle non disponible. Veuillez réessayer plus tard."
            )
        
        # Prépare les données pour le modèle
        # Le modèle attend un tableau 2D [[ca_total, nb_commandes]]
        features = np.array([[input_data.ca_total, input_data.nb_commandes]])
        
        logger.info(f"Features préparées: {features}")
        
        # Fait la prédiction
        prediction = model.predict(features)
        
        # Récupère la valeur prédite (premier élément du tableau)
        prediction_value = float(prediction[0])
        log_prediction_to_csv(
            ca_total=input_data.ca_total,
            nb_commandes=input_data.nb_commandes,
            prediction=prediction_value,
            model_version="1.0"
        )   
        
        logger.info(f"Prédiction générée: {prediction_value}")
        
        # Prépare la réponse
        response = PredictionOutput(
            prediction_ca_jour_suivant=prediction_value,
            inputs=input_data,
            timestamp=datetime.now().isoformat(),
            model_version="1.0"
        )
        
        # Journalisation de la réponse
        logger.info(f"Réponse envoyée: {response}")
        
        return response
        
    except ValueError as e:
        logger.error(f"Erreur de valeur: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Données invalides: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne: {str(e)}"
        ) 
@app.get("/model/info")
async def model_info():
    """
    Endpoint pour obtenir des informations sur le modèle chargé
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        info = {
            "model_type": str(type(model)),
            "model_version": "1.0",
            "features_used": ["ca_total", "nb_commandes"],
            "model_params": model.get_params() if hasattr(model, 'get_params') else {},
            "n_estimators": model.n_estimators if hasattr(model, 'n_estimators') else None,
            "loaded_at": "À déterminer"  # Vous pourriez stocker cette info au chargement
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des infos du modèle: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne")
if __name__ == "__main__":
    """
    Point d'entrée pour exécuter l'API localement
    """
    import uvicorn
    
    # Configuration du serveur
    host = "0.0.0.0"  # Écoute sur toutes les interfaces
    port = 8000        # Port par défaut
    
    logger.info(f"Démarrage du serveur sur http://{host}:{port}")
    
    # Lancement du serveur
    uvicorn.run(
        "src.api:app",     # Module:application
        host=host,
        port=port,
        reload=True,    # Recharge automatiquement en développement
        log_level="info"
    )
    #fonction de journalisation 
def log_prediction_to_csv(
    ca_total: float,
    nb_commandes: int,
    prediction: float,
    model_version: str
):
    """
    Sauvegarde chaque prédiction dans un fichier CSV pour le monitoring
    """
    file_exists = PREDICTION_LOG_PATH.exists()

    with open(PREDICTION_LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Écriture de l'en-tête si le fichier n'existe pas
        if not file_exists:
            writer.writerow([
                "timestamp",
                "ca_total",
                "nb_commandes",
                "prediction",
                "model_version"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            ca_total,
            nb_commandes,
            prediction,
            model_version
        ])
