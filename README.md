# Mini-projet MLOps & DevSecOps

**Prédiction du chiffre d’affaires journalier**
## Badge

[![CI/CD Pipeline](https://github.com/azizilahi2002-afk/MlOps-project/actions/workflows/ci.yml/badge.svg)](https://github.com/azizilahi2002-afk/MlOps-project/actions/workflows/ci.yml)



## Description

Ce projet est un mini-projet pédagogique MLOps & DevSecOps visant à couvrir l’ensemble du cycle de vie d’un modèle de Machine Learning en production : préparation des données, entraînement du modèle, déploiement via une API FastAPI, monitoring des prédictions et bonnes pratiques de qualité et de sécurité.

## Objectifs

* Mettre en œuvre un pipeline complet **Data → Modèle → API → Monitoring**
* Comprendre les principes du **MLOps**
* Appliquer des pratiques **DevSecOps** (tests, hooks, audit)
* Surveiller le comportement d’un modèle en production

## Structure du projet

```
mini-mlops-project/
├── data/
│   ├── raw/ventes_ecommerce.csv
│   ├── processed/features_daily.parquet
│   └── predictions_log.csv
├── models/random_forest_model.pkl
├── notebooks/
│   ├── exploration.ipynb
│   └── monitoring.ipynb
├── reports/
│   ├── predictions_time_series.png
│   └── predictions_histogram.png
├── src/
│   ├── data_preparation.py
│   ├── train.py
│   └── api.py
├── tests/test_api.py
├── .pre-commit-config.yaml
├── requirements.txt
└── README.md
```

## Données

Le jeu de données e-commerce brut est agrégé par jour pour produire des features quotidiennes :

* `ca_total` : chiffre d’affaires total
* `nb_commandes` : nombre de commandes
* `label` : CA du jour suivant

Une colonne de date artificielle est utilisée pour simuler un contexte réel.

## Modèle

* Modèle : **RandomForestRegressor**
* Features : `ca_total`, `nb_commandes`
* Split temporel : 80 % train / 20 % test
* Métriques : MAE, RMSE, R²

Le modèle entraîné est sauvegardé dans le dossier `models/`.

## API de prédiction

* Technologie : **FastAPI**
* Serveur : **Uvicorn**

### Endpoints

| Endpoint   | Méthode | Description                      |
| ---------- | ------- | -------------------------------- |
| `/`        | GET     | Accueil                          |
| `/health`  | GET     | État de l’API                    |
| `/predict` | POST    | Prédiction du CA du jour suivant |
| `/docs`    | GET     | Documentation Swagger            |

### Exemple de requête

```json
{
  "ca_total": 15000.0,
  "nb_commandes": 320
}
```

### Exemple de réponse

```json
{
  "prediction_ca_jour_suivant": 16234.56,
  "inputs": {"ca_total": 15000.0, "nb_commandes": 320},
  "timestamp": "2025-12-17T11:20:00",
  "model_version": "1.0"
}
```

## Monitoring

Chaque appel à `/predict` est journalisé dans `data/predictions_log.csv` (timestamp, features, prédiction, version du modèle).

Le notebook `notebooks/monitoring.ipynb` permet :

* l’analyse temporelle des prédictions,
* l’étude de leur distribution,
* l’identification d’anomalies potentielles.

Les graphiques sont exportés dans `reports/`.

## Qualité & Sécurité

* **Tests unitaires** avec `pytest`
* **Hooks pre-commit** : black, ruff, detect-secrets, nbstripout
* **Audit de dépendances** avec `pip-audit`

## Installation et exécution

### 1. Installation des dépendances

Créer (optionnel) et activer un environnement virtuel, puis installer les dépendances :

```bash
pip install -r requirements.txt
```

Vérifier que les outils principaux sont disponibles :

* Python 3.10+
* FastAPI
* Uvicorn
* scikit-learn

---

### 2. Lancement du pipeline de données

Le pipeline ETL prépare les données brutes et génère les features quotidiennes.

```bash
python src/data_preparation.py
```

Sortie attendue :

* Fichier agrégé `data/processed/features_daily.parquet`
* Colonnes : `date`, `ca_total`, `nb_commandes`, `label`

---

### 3. Entraînement du modèle

L’entraînement utilise un split temporel et sauvegarde le modèle entraîné.

```bash
python src/train.py
```

Sortie attendue :

* Modèle sauvegardé dans `models/random_forest_model.pkl`
* Affichage des métriques (MAE, RMSE, R²)

---

### 4. Lancement de l’API

Démarrer l’API FastAPI avec Uvicorn depuis la racine du projet :

```bash
uvicorn src.api:app --reload --port 8000
```

Accès :

* API : [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Health check : [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
* Documentation Swagger : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 5. Utilisation du monitoring

Chaque appel à l’endpoint `/predict` est automatiquement journalisé dans :

```
data/predictions_log.csv
```

Contenu du journal :

* `timestamp` : date et heure de l’appel
* `ca_total`, `nb_commandes` : features d’entrée
* `prediction` : valeur prédite
* `model_version` : version du modèle

Pour analyser le comportement du modèle en production :

1. Ouvrir le notebook `notebooks/monitoring.ipynb`
2. Charger le fichier `predictions_log.csv`
3. Générer les visualisations (évolution temporelle, distribution, relations)
4. Exporter les graphiques dans le dossier `reports/`

Ces analyses permettent d’identifier des anomalies potentielles et d’évaluer la stabilité du modèle.

Accès Swagger : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

```bash
pip install -r requirements.txt
python src/data_preparation.py
python src/train.py
uvicorn src.api:app --reload --port 8000
```

Accès Swagger : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Améliorations possibles

* Enrichissement des features (saisonnalité)
* Tracking d’expériences (MLflow)
* CI/CD (GitHub Actions)
* Dockerisation
* Monitoring avancé de dérive

## Conclusion

Ce mini-projet fournit une base pédagogique complète pour appréhender les concepts MLOps et DevSecOps dans un contexte réaliste, de la donnée au monitoring en production.
