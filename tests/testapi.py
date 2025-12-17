import requests
import json

# URL de base de l'API
BASE_URL = "http://localhost:5000"

def test_health():
    """Test de l'endpoint /health"""
    print("=== Test de /health ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_predict():
    """Test de l'endpoint /predict"""
    print("=== Test de /predict ===")
    
    # Données de test
    test_data = {
        "ca_total": 15000.0,
        "nb_commandes": 320
    }
    
    # Headers
    headers = {
        "Content-Type": "application/json"
    }
    
    # Envoi de la requête
    response = requests.post(
        f"{BASE_URL}/predict",
        data=json.dumps(test_data),
        headers=headers
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Prédiction: {result['prediction_ca_jour_suivant']}")
        print(f"Inputs: {result['inputs']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_model_info():
    """Test de l'endpoint /model/info"""
    print("=== Test de /model/info ===")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

if __name__ == "__main__":
    print("Démarrage des tests de l'API...\n")
    
    try:
        # Test 1: Endpoint de santé
        test_health()
        
        # Test 2: Informations du modèle
        test_model_info()
        
        # Test 3: Prédiction
        test_predict()
        
        # Test supplémentaire avec d'autres valeurs
        print("=== Test avec différentes valeurs ===")
        test_cases = [
            {"ca_total": 10000.0, "nb_commandes": 200},
            {"ca_total": 20000.0, "nb_commandes": 400},
            {"ca_total": 5000.0, "nb_commandes": 100}
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest case {i}: {test_case}")
            response = requests.post(
                f"{BASE_URL}/predict",
                json=test_case
            )
            if response.status_code == 200:
                result = response.json()
                print(f"  Prédiction: {result['prediction_ca_jour_suivant']:.2f}")
            else:
                print(f"  Erreur: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("Erreur: Impossible de se connecter à l'API")
        print("Assurez-vous que l'API est en cours d'exécution (uvicorn src.api:app --reload)")
    except Exception as e:
        print(f"Erreur inattendue: {e}")