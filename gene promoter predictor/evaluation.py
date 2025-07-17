# evaluation_log.py
import logging
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import datetime

# Configure logging
logging.basicConfig(
    filename='results_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def evaluate_and_log(model, X_test, y_test, sample_df):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logging.info("===== Evaluation Results =====")
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info("Classification Report:\n%s", report)
    logging.info("Sample Features Used (first 5 rows):\n%s",
                 sample_df[['gc_content', 'at_content', 'dinuc_repeats', 'has_tata', 'entropy', 'length']].head().to_string())
    logging.info("============================\n")

    print("Evaluation complete. Results saved to results_log.log")