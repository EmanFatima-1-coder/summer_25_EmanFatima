import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    mean_absolute_error, mean_squared_error
)
import numpy as np

def model(df, scaler_type='standard', encoding_type='label'):

    # Step 1: Separate features and labels
    X = df.drop(columns=['labels'])
    y = df['labels']

    # Step 2: Encode labels
    if encoding_type == 'label':
        le = LabelEncoder()
        y = le.fit_transform(y)
    elif encoding_type == 'onehot':
        ohe = OneHotEncoder(sparse=False)
        y = ohe.fit_transform(y.values.reshape(-1, 1))
    elif encoding_type == 'ordinal':
        oe = OrdinalEncoder()
        y = oe.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Step 3: Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Feature scaling
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'maxabs':
        scaler = MaxAbsScaler()
    else:
        scaler = StandardScaler()  # default

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 5: Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Step 6: Predict
    y_pred = model.predict(X_test_scaled)

    # Step 7: Evaluation (classification)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Step 8: Error Metrics (apply only if y is numeric)
    try:
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = None  # R² score not valid for categorical
        adj_r2 = None
    except:
        mae = mse = rmse = r2 = adj_r2 = "Not Applicable for Categorical Targets"

    return {
        'model': model,
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Adjusted_R2': adj_r2
    }
