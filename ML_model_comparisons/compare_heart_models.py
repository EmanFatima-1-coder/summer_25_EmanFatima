import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

# ----------------------------
# Step 1: Load the dataset
# ----------------------------
file_path = 'heart.csv'  # Make sure heart.csv is in the same folder
df = pd.read_csv(file_path)

# ----------------------------
# Step 2: Clean & prepare data
# ----------------------------
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df = df.astype(float)
df['condition'] = df['condition'].apply(lambda x: 1 if x > 0 else 0)

# ----------------------------
# Step 3: Split and scale
# ----------------------------
X = df.drop('condition', axis=1)
y = df['condition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Step 4: Define ML models
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Bagging Classifier": BaggingClassifier(random_state=42),
    "Extra Tree": ExtraTreeClassifier(random_state=42),
    "Ridge Classifier": RidgeClassifier(),
    "Naive Bayes": GaussianNB()
}

# ----------------------------
# Step 5: Train & evaluate
# ----------------------------
results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }

# Convert to DataFrame and sort by ROC-AUC
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by='ROC-AUC', ascending=False)

# ----------------------------
# Step 6: Show results
# ----------------------------
print("\n--- Model Performance Comparison ---")
print(results_df)

# ----------------------------
# Step 7: Visualizations
# ----------------------------

# Bar plot for F1-score comparison across models
plt.figure(figsize=(10, 6))
sns.barplot(x=results_df["F1-Score"], y=results_df.index, palette="viridis")
plt.title("F1-Score Comparison Across Models")
plt.xlabel("F1-Score")
plt.ylabel("Models")
plt.grid(True)
plt.tight_layout()
plt.show()

# Only show the confusion matrix for the best model
best_model_name = results_df.index[0]
best_model = models[best_model_name]

print(f"\n✅ Best Performing Model: {best_model_name}")
y_pred_best = best_model.predict(X_test_scaled)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_best)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.grid(False)
plt.show()

# ----------------------------
# Step 8: Save results to CSV
# ----------------------------
results_df.to_csv("model_comparison_results.csv")
print("📁 Results saved to 'model_comparison_results.csv'")
