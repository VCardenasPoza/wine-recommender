import pandas as pd


def clean_wine_dataframe(df: pd.DataFrame,n_cepas: int) -> pd.DataFrame:
    """
    Clean raw wine dataframe keeping only relevant columns
    and rows.
    """
    df = df.copy()

    columns_to_keep = [
        "id",
        "name",
        "precio",
        "puntaje",
        "tipo",
        "cepa",
        "mezcla",
        "cosecha",
        "valle",
        "nota_cata",
        "grados",
        "frescor",
        "cuerpo",
        "dulzor",
        "astringencia",
        "m_carnesrojas",
        "m_aves",
        "m_quesos",
        "m_carnescaza",
        "m_pescados",
        "m_comidachilena",
        "m_cerdo",
        "m_cordero",
        "m_pasta",
        "m_crustaceos",
        "m_mariscos",
        "m_comidaoriental",
        "m_postres",
    ]

    df = df[columns_to_keep]

    df = df.dropna(subset=["nota_cata"])
    df.drop_duplicates(subset=['name'],inplace=True)
    df = df[df["precio"]>5000]
    df = df[df["grados"]> 1.0]

    total_variety  = df["cepa"].value_counts()
    varietys_to_study = total_variety[total_variety.index != "Sin Informacion"].index[:n_cepas]
    
    df = df[df["cepa"].isin(varietys_to_study)]

    return df
