# Tous les imports doivent être au début
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.inspection import permutation_importance
import joblib
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_classification

warnings.filterwarnings("ignore")

# ... Le reste du code sans nouveaux imports ...

df = pd.read_parquet("data/processed/features_daily.parquet")

X = df[["ca_total", "nb_commandes"]]
Y = df["label"]
split_idx = int(len(df) * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = Y.iloc[:split_idx]
y_test = Y.iloc[split_idx:]
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

print("\n ...")
model.fit(X_train, y_train)
print("entrainement terminé")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\n=== RÉSULTATS D'ÉVALUATION ===")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.3f}")


model_path = Path("models/random_forest_model.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_path)
print(f"\nModèle sauvegardé: {model_path}")

# Exemple avec un dataset
# Remplacez par votre propre dataset

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
)

# Split train/test stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42  # Garde les proportions de classes
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Distribution y_train: {pd.Series(y_train).value_counts()}")

# Modèle avec poids équilibrés pour classes déséquilibrées
model = LogisticRegression(max_iter=200, class_weight="balanced", random_state=42)

# K-Fold stratifié
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Scores F1 sur chaque fold
f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

print(f"F1 scores par fold: {f1_scores}")
print(f"F1 CV moyen: {f1_scores.mean():.3f} (+/- {f1_scores.std():.3f})")

# Entraîner sur tout le train set
model.fit(X_train, y_train)

# Prédictions sur le test set
y_pred = model.predict(X_test)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion:")
print(cm)
print(f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

# Rapport complet
print("\nRapport de classification:")
print(classification_report(y_test, y_pred, digits=3))

# Probabilités prédites (pour la classe positive)
y_score = model.predict_proba(X_test)[:, 1]

# Courbe ROC
fpr, tpr, thresholds_roc = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Courbe Precision-Recall
prec, rec, thresholds_pr = precision_recall_curve(y_test, y_score)
pr_auc = auc(rec, prec)

print(f"ROC-AUC: {roc_auc:.3f}")
print(f"PR-AUC: {pr_auc:.3f}")

# Visualisation
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"ROC-AUC={roc_auc:.3f}", linewidth=2)
plt.plot([0, 1], [0, 1], "--", color="gray", label="Aléatoire")
plt.xlabel("Taux de Faux Positifs (FPR)")
plt.ylabel("Taux de Vrais Positifs (TPR)")
plt.title("Courbe ROC")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(rec, prec, label=f"PR-AUC={pr_auc:.3f}", color="green", linewidth=2)
plt.xlabel("Recall (Sensibilité)")
plt.ylabel("Precision")
plt.title("Courbe Precision-Recall")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Définir les coûts métier
cost_fn = 200  # Coût d'un Faux Négatif (ex: fraude manquée)
cost_fp = 5  # Coût d'un Faux Positif (ex: fausse alerte)

# Tester différents seuils
thresholds = np.linspace(0.0, 1.0, 201)
y_score = model.predict_proba(X_test)[:, 1]

results = []
for thr in thresholds:
    y_thr = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_thr).ravel()
    total_cost = fn * cost_fn + fp * cost_fp
    results.append(
        {"threshold": thr, "cost": total_cost, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    )

# Trouver le seuil optimal
best = min(results, key=lambda x: x["cost"])
print(f"Seuil optimal: {best['threshold']:.3f}")
print(f"Coût total: {best['cost']}")
print(f"TP={best['tp']}, FP={best['fp']}, FN={best['fn']}, TN={best['tn']}")

# Visualiser
costs = [r["cost"] for r in results]
thrs = [r["threshold"] for r in results]
plt.figure(figsize=(10, 4))
plt.plot(thrs, costs, linewidth=2)
plt.axvline(
    best["threshold"],
    color="red",
    linestyle="--",
    label=f"Optimal: {best['threshold']:.3f}",
)
plt.xlabel("Seuil de décision")
plt.ylabel("Coût total")
plt.title("Optimisation du seuil selon les coûts métier")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Matrice de confusion annotée
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Classe 0", "Classe 1"],
    yticklabels=["Classe 0", "Classe 1"],
)
plt.ylabel("Vraie classe")
plt.xlabel("Classe prédite")
plt.title("Matrice de confusion")
plt.show()

# Identifier les cas difficiles
errors_idx = np.where(y_pred != y_test)[0]
print(f"Nombre d'erreurs: {len(errors_idx)}")

# Exemples de FP et FN
fp_idx = np.where((y_pred == 1) & (y_test == 0))[0]
fn_idx = np.where((y_pred == 0) & (y_test == 1))[0]

print(f"\nFaux Positifs: {len(fp_idx)} cas")
print(f"Faux Négatifs: {len(fn_idx)} cas")

# Analyser les scores de probabilité des erreurs
print("\nScores moyens:")
print(f"  FP: {y_score[fp_idx].mean():.3f}")
print(f"  FN: {y_score[fn_idx].mean():.3f}")

# Importance par permutation
result = permutation_importance(
    model, X_test, y_test, scoring="f1", n_repeats=10, random_state=42
)

# Top 10 features
sorted_idx = result.importances_mean.argsort()[::-1][:10]

plt.figure(figsize=(10, 6))
plt.barh(
    range(len(sorted_idx)),
    result.importances_mean[sorted_idx],
    xerr=result.importances_std[sorted_idx],
)
plt.yticks(range(len(sorted_idx)), [f"Feature {i}" for i in sorted_idx])
plt.xlabel("Importance (baisse de F1 si permutée)")
plt.title("Top 10 Features par Permutation Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("Top features:", sorted_idx)
