import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from src.utils import info, clean_column_names, handle_infinite

def load_cicids_data(data_dir: str) -> pd.DataFrame:
    info(f"Loading CSVs from {data_dir}")
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            info(f"Skipping {f} due to {e}")
    data = pd.concat(dfs, ignore_index=True)
    info(f"Total rows: {data.shape[0]}, columns: {data.shape[1]}")
    return data

def preprocess_data(df: pd.DataFrame):
    info("Preprocessing data...")
    df = clean_column_names(df)
    df = handle_infinite(df)

    # Drop non-numeric & metadata columns
    drop_cols = [c for c in df.columns if 'Flow_ID' in c or 'IP' in c or 'Timestamp' in c]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Fill NaNs
    for c in df.columns:
        if df[c].dtype.kind in 'biufc':
            df[c].fillna(df[c].median())
        else:
            df[c].fillna(df[c].mode().iloc[0])

    # Label column
    target_col = 'Label' if 'Label' in df.columns else df.columns[-1]
    df['binary_label'] = df[target_col].apply(lambda x: 'Benign' if str(x).lower() == 'benign' else 'Attack')

    X = df.select_dtypes(include=['number'])
    y = (df['binary_label'] == 'Attack').astype(int)

    info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    info("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Replace infinite or NaN values that may have appeared after scaling
    import numpy as np
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    info("Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    info(f"Balanced train size: {X_train.shape}")


    return X_train, X_test, y_train, y_test, scaler
