# main.py
from features import load_and_prepare_data
from train_model import ANNModel, train_model, plot_training_loss, plot_confusion_matrix, plot_roc_curve
from evaluate_model import evaluate_model
import numpy as np
import torch

if __name__ == "__main__":
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data("annotated_snv_data.csv")
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("Building ANN model...")
    model = ANNModel(input_dim=X_train.shape[1])

    print("Training model...")
    model, epoch_losses = train_model(model, X_train, y_train)

    print("Visualizing training loss...")
    plot_training_loss(epoch_losses)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("Generating predictions for additional visualizations...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        y_pred_probs = outputs.cpu().numpy().flatten()
        y_pred = (y_pred_probs > 0.5).astype(int)

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_probs)
    
    