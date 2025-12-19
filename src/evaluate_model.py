# src/evaluate_model.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import joblib
import os

# --- Configuration ---
MODEL_PATH = "models/random_forest_model.pkl"
DATA_PATH = "data/processed/features_daily.parquet"


def load_data_and_model():
    """Charge les données et le modèle entraîné."""
    print("Chargement des données et du modèle...")
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values("date")

    X = df[["ca_total", "nb_commandes"]]
    y = df["label"]

    model = joblib.load(MODEL_PATH)
    return X, y, model


def evaluate_model_with_cv():
    """
    Évalue le modèle en utilisant une validation croisée temporelle.
    C'est l'équivalent du "K-Fold stratifié" pour les séries temporelles.
    """
    X, y, model = load_data_and_model()

    # TimeSeriesSplit est la meilleure pratique pour les données temporelles
    tscv = TimeSeriesSplit(n_splits=5)

    # On évalue sur plusieurs métriques, comme demandé dans le TP3
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }

    results = {}
    for name, score in scoring.items():
        # cross_val_score retourne des scores négatifs pour les erreurs
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring=score)
        if name in ["MAE", "RMSE"]:
            # On remet en positif pour l'interprétation
            results[name] = -cv_scores.mean()
        else:
            results[name] = cv_scores.mean()

    print("\n--- Résultats de la Validation Croisée (TimeSeriesSplit) ---")
    for metric, value in results.items():
        print(f"{metric}: {value:.2f}")

    return results


# Ajoutez cette fonction dans le fichier src/evaluate_model.py


def plot_predictions_vs_actual():
    """Génère un graphique des prédictions contre les valeurs réelles."""
    X, y, model = load_data_and_model()

    # On prédit sur l'ensemble des données pour la visualisation
    predictions = model.predict(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(y, predictions, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "--r", linewidth=2)  # Ligne y=x
    plt.xlabel("Valeurs Réelles (CA du jour suivant)")
    plt.ylabel("Prédictions du Modèle")
    plt.title("Prédictions vs Valeurs Réelles")
    plt.grid(True)

    # Sauvegarder le graphique
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/predictions_vs_actual.png")
    print(
        "\nGraphique 'predictions_vs_actual.png' sauvegardé dans le dossier 'reports'."
    )
    plt.show()


# Ajoutez un appel à cette fonction dans le bloc if __name__ == "__main__":
# if __name__ == "__main__":
#     evaluate_model_with_cv()
#     plot_predictions_vs_actual()


# Ajoutez cette fonction dans le fichier src/evaluate_model.py


def plot_feature_importance():
    """Affiche et sauvegarde l'importance des features du modèle."""
    X, y, model = load_data_and_model()

    importances = model.feature_importances_
    feature_names = X.columns

    # Créer un DataFrame pour plus de clarté
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    print("\n--- Importance des Features ---")
    print(feature_importance_df)

    # Visualisation
    plt.figure(figsize=(8, 5))
    sns.barplot(x="importance", y="feature", data=feature_importance_df)
    plt.title("Importance des Features")
    plt.xlabel("Importance")
    plt.tight_layout()

    # Sauvegarder le graphique
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/feature_importance.png")
    print("\nGraphique 'feature_importance.png' sauvegardé dans le dossier 'reports'.")
    plt.show()


# Mettez à jour le bloc if __name__ == "__main__" pour l'inclure :
# if __name__ == "__main__":
#     evaluate_model_with_cv()
#     plot_predictions_vs_actual()
#     plot_feature_importance()

if __name__ == "__main__":
    # Pour lancer l'évaluation directement depuis le terminal
    # python src/evaluate_model.py
    evaluate_model_with_cv()
    plot_predictions_vs_actual()
    plot_feature_importance()
