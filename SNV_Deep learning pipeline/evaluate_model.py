import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Ensure inputs are NumPy arrays
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = np.array(y_test).astype(int)

    with torch.no_grad():
        outputs = model(X_test)
        y_pred_probs = outputs.cpu().numpy().flatten()
        y_pred = (y_pred_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
