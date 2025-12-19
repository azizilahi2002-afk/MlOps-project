import os
import runpy

import pandas as pd
import pytest

SCRIPT_PATH = "src/data_preparation.py"
OUT_PATH = "data/processed/features_daily.parquet"


@pytest.fixture(scope="module")
def run_preprocessing_script():
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    runpy.run_path(SCRIPT_PATH)

    yield OUT_PATH

    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)


def test_output_file_created(run_preprocessing_script):
    assert os.path.exists(run_preprocessing_script)


def test_output_schema(run_preprocessing_script):
    df = pd.read_parquet(run_preprocessing_script)
    expected_columns = ["date", "ca_total", "nb_commandes", "label"]

    assert all(col in df.columns for col in expected_columns)
    assert len(df.columns) == len(expected_columns)


def test_label_is_correctly_shifted(run_preprocessing_script):
    df = pd.read_parquet(run_preprocessing_script)

    if len(df) > 1:
        assert df.loc[0, "label"] == df.loc[1, "ca_total"]


def test_no_missing_values(run_preprocessing_script):
    df = pd.read_parquet(run_preprocessing_script)
    assert df.isnull().sum().sum() == 0
