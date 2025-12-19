import os
import runpy
from unittest.mock import patch

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

# Chemins utilisés par le script et les tests
SCRIPT_PATH = "src/train.py"
MODEL_OUT_PATH = "models/random_forest_model.pkl"
DATA_IN_PATH = "data/processed/features_daily.parquet"


@pytest.fixture(scope="module")
def setup_and_teardown_for_model_test():
    """
    Prépare l'environnement pour le test du modèle et nettoie après.
    """
    # --- SETUP ---
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    dummy_data = {
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "ca_total": [100, 150, 120],
        "nb_commandes": [10, 15, 12],
        "label": [150, 120, 110],
    }
    pd.DataFrame(dummy_data).to_parquet(DATA_IN_PATH, index=False)

    if os.path.exists(MODEL_OUT_PATH):
        os.remove(MODEL_OUT_PATH)

    with patch("matplotlib.pyplot.show"):
        runpy.run_path(SCRIPT_PATH)

    yield

    # --- TEARDOWN ---
    if os.path.exists(DATA_IN_PATH):
        os.remove(DATA_IN_PATH)
    if os.path.exists(MODEL_OUT_PATH):
        os.remove(MODEL_OUT_PATH)


def test_model_file_is_created(setup_and_teardown_for_model_test):
    assert os.path.exists(MODEL_OUT_PATH)


def test_saved_model_is_correct_type(setup_and_teardown_for_model_test):
    loaded_model = joblib.load(MODEL_OUT_PATH)
    assert isinstance(loaded_model, RandomForestRegressor)
