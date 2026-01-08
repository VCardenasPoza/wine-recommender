from pathlib import Path
import pandas as pd

from src.data.clean_dataframe import clean_wine_dataframe

from src.utils.io import load_csv, save_csv


RAW_PATH = Path("data/raw/wines_raw.csv")
OUT_PATH = Path("data/processed/wines_clean.csv")


def make_dataset(
    n_cepas:int,
    raw_path: Path = RAW_PATH,
    out_path: Path = OUT_PATH
) -> pd.DataFrame:
    """
    Build clean dataframe from raw data
    """

    df = load_csv(raw_path)

    df = clean_wine_dataframe(df,n_cepas)

    save_csv(df, out_path)

    return df


if __name__ == "__main__":
    make_dataset()
