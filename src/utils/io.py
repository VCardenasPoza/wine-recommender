from pathlib import Path
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    """
    Load dataframe from path in the project.
    """
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Save dataframe in a determinate path in the project.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
