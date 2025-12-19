import pandas as pd
import os

RAW_PATH = "data/raw/ventes_ecommerce.csv"
OUT_PATH = "data/processed/features_daily.parquet"

print("1) Chargement du fichier brut...")
df = pd.read_csv(RAW_PATH)

# 2) Création d'une colonne de date artificielle sur exactement 30 jours
if "date" not in df.columns:
    print("Aucune colonne 'date' trouvée. Création artificielle...")

    # On génère exactement len(df) dates réparties uniformément sur 30 jours
    # Du 2024-01-01 00:00 au 2024-01-30 23:59
    date_range = pd.date_range(
        start="2024-01-01",
        end="2024-01-30 23:59:59",
        periods=len(df),  # répartir uniformément les 10000 lignes sur 30 jours
    )

    # On ne garde que la partie date (sans heure)
    df["date"] = date_range.date

# Conversion en datetime
df["date"] = pd.to_datetime(df["date"])

# 3) Détection de la colonne de prix
price_col = None
for col in df.columns:
    if col.lower().replace(" ", "_") in ["purchase_price", "price", "montant"]:
        price_col = col
        break

if price_col is None:
    raise ValueError(
        "Impossible de trouver la colonne du prix (Purchase Price / Price / Montant)"
    )

print(f"Colonne de prix détectée : {price_col}")
print("3) Calcul des indicateurs quotidiens...")

# Agrégation quotidienne
daily = (
    df.groupby("date")
    .agg(ca_total=(price_col, "sum"), nb_commandes=(price_col, "count"))
    .reset_index()
)

# 4) Label = CA du jour suivant
daily["label"] = daily["ca_total"].shift(-1)

# Supprime la dernière ligne (pas de label)
daily = daily.dropna().reset_index(drop=True)

# 5) Sauvegarde
os.makedirs("data/processed", exist_ok=True)
daily.to_parquet(OUT_PATH, index=False)

print("\nPipeline terminé.")
print(f"Fichier sauvegardé : {OUT_PATH}")
print("Nombre de lignes :", len(daily))
print("Colonnes :", list(daily.columns))
print("Dates min/max :", daily["date"].min(), "->", daily["date"].max())
