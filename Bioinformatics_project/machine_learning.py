import pandas as pd
import numpy as np
from Bio.Seq import Seq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from Bio.SeqUtils import molecular_weight, MeltingTemp as mt
import math

# GC Content Calculator
def calculate_gc(seq):
    seq = seq.upper()
    G = seq.count('G')
    C = seq.count('C')
    N = len(seq)
    return (G + C) / N if N > 0 else 0

# Feature Extraction
def extract_features(seq):
    seq = seq.upper()
    A = seq.count('A')
    C = seq.count('C')
    G = seq.count('G')
    T = seq.count('T')
    N = len(seq)

    gc_content = calculate_gc(seq)
    mol_weight = molecular_weight(seq, seq_type='DNA')
    melting_temp = mt.Tm_Wallace(seq)

    gc_skew = (G - C) / (G + C) if (G + C) != 0 else 0
    at_skew = (A - T) / (A + T) if (A + T) != 0 else 0

    probs = [A/N, C/N, G/N, T/N] if N > 0 else [0, 0, 0, 0]
    entropy = -sum([p * math.log2(p) for p in probs if p > 0])

    return pd.Series([A, C, G, T, N, gc_content, mol_weight, melting_temp, gc_skew, at_skew, entropy],
                     index=['A', 'C', 'G', 'T', 'Length', 'GC_Content', 'Mol_Weight',
                            'Melting_Temp', 'GC_Skew', 'AT_Skew', 'Shanon_Entropy'])

# Main Pipeline
def main():
    # Load data
    data = pd.read_csv('dna_sequences.csv')  # Must contain 'sequence' and 'label' columns

    # Extract features
    features = data['sequence'].apply(extract_features)
    features = pd.concat([data['label'], features], axis=1)

    print("\nPreview of extracted features:")
    print(features.head())

    # Prepare X and y
    X = features.drop('label', axis=1)
    y = features['label']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    print(y_pred)
    # Evaluation
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    main()
