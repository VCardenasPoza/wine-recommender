from pathlib import Path
import pandas as pd

from src.data.clean_dataframe import clean_dataframe

from src.utils.io import load_csv, save_csv


RAW_PATH = Path("data/raw/wines.csv")
OUT_PATH = Path("data/processed/wines_clean.parquet")


def make_dataset(
    raw_path: Path = RAW_PATH,
    out_path: Path = OUT_PATH
) -> pd.DataFrame:
    """
    Build clean dataframe 
    """

    df = load_csv(raw_path)

    df = clean_dataframe(df)

    save_csv(df, out_path)

    return df


if __name__ == "__main__":
    make_dataset()
