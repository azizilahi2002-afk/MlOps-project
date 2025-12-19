# Importer les librairies
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Chemins des fichiers
DATA_PATH = "data/processed/features_daily.parquet"
MODEL_PATH = "models/model_latest.pkl"

# Charger les données
df = pd.read_parquet(DATA_PATH)

# Séparer features et cible
X = df[["ca_total", "nb_commandes"]]
y = df["label"]

# Vérifier les premières lignes
print(df.head())
# Split chronologique 80% train / 20% test
train_size = int(len(df)) * 0.8
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Vérifier les tailles
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
# Créer le modèle
model = RandomForestRegressor(n_estimators=200, random_state=42)

# Entraîner le modèle
model.fit(X_train, y_train)
# Prédictions sur le test
y_pred = model.predict(X_test)

# Calculer les métriques
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE :", mean_squared_error(y_test, y_pred, squared=False))
print("R2 :", r2_score(y_test, y_pred))
# Sauvegarder le modèle
joblib.dump(model, MODEL_PATH)
print("📌 Modèle sauvegardé :", MODEL_PATH)
# Feature importance
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.sort_values().plot(kind="barh")
plt.title("Feature Importance")
plt.show()
