import os
import pandas as pd
from src.data_ingestion.ingestion import ingest_and_validate


def test_ingest_and_validate(tmp_path):
    raw_path = "data/raw/transactions_fraud.csv"
    interim_path = tmp_path / "interim.csv"

    df = ingest_and_validate(raw_path, str(interim_path))

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "is_fraud" in df.columns
    assert os.path.exists(interim_path)