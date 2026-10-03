import pandas as pd
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw dataset path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "wfp_food_prices_nga.csv"


def load_data():
    """Load the raw WFP Nigeria food price dataset."""
    df = pd.read_csv(DATA_PATH)
    return df


def main():
    df = load_data()

    print("\n=== Dataset Loaded Successfully ===")
    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    print("\n=== UNIQUE STATES ===")
    print(f"Total states: {df['admin1'].nunique()}")
    print(sorted(df["admin1"].unique()))

    print("\n=== UNIQUE COMMODITIES ===")
    print(f"Total commodities: {df['commodity'].nunique()}")
    print(sorted(df["commodity"].unique()))


if __name__ == "__main__":
    main()