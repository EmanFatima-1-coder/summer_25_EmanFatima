import logging
from extract_features import protein_feature_analysis, synthetic_labels
from model import model

# Configure logging
logging.basicConfig(level=logging.INFO, filename='output.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    fasta_file = r"C:\Users\pmyls\OneDrive\Desktop\project\proteins.fasta"
    logging.info("Parsing and analyzing FASTA file...")
    data_frame = protein_feature_analysis(fasta_file)

    logging.info("Features extracted successfully. Assigning labels...")
    df = synthetic_labels(data_frame)

    logging.info("Running classification model...")
    results = model(df, scaler_type='standard', encoding_type='label')

    logging.info(f"Model: {results['model']}")
    logging.info(f"Accuracy: {results['accuracy']}")
    logging.info(f"Classification Report:\n{results['classification_report']}")
    logging.info(f"Confusion Matrix:\n{results['confusion_matrix']}")
    logging.info(f"MAE: {results['MAE']}")
    logging.info(f"MSE: {results['MSE']}")
    logging.info(f"RMSE: {results['RMSE']}")
    logging.info(f"R2: {results['R2']}")
    logging.info(f"Adjusted R2: {results['Adjusted_R2']}")

if __name__ == "__main__":
    main()
