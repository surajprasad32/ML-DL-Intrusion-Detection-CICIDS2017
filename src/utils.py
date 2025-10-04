import pandas as pd
import numpy as np

def info(msg: str):
    print(f"[INFO] {msg}")

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    return df

def handle_infinite(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    return df
