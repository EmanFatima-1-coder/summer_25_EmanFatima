# features.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['impact', 'sift', 'polyphen', 'consequence', 'Label'])

    X = df[['impact', 'sift', 'polyphen', 'consequence']]
    y = df['Label'].astype(int)

    encoder = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown='ignore'), X.columns)],
        remainder='drop'
    )
    X_encoded = encoder.fit_transform(X)

    return train_test_split(X_encoded, y, test_size=0.2, random_state=42)
