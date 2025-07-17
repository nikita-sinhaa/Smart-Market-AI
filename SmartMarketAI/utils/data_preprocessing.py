import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)

    # Encode categorical columns if needed
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category').cat.codes

    return df

def preprocess(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
