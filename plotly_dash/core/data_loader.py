import pandas as pd
import os


def load_sample_data(n=50000):

    path = "data/interim/transactions_clean.csv"
    if not os.path.exists(path):
        path = "data/raw/transactions_fraud.csv"

    df = pd.read_csv(path)

    if len(df) > n:
        df = df.sample(n, random_state=42)

    return df