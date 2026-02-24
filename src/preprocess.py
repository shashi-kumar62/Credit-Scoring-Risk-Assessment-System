import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove null values
    df = df.dropna()

    # Convert target
    df["kredit"] = df["kredit"].replace({1: 1, 2: 0})

    # Feature engineering
    df["credit_per_month"] = df["hoehe"] / df["laufzeit"]

    return df