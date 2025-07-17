# Gene Promoter Predictor

This project is a machine learning pipeline for predicting gene promoter strength (strong/weak) from DNA sequence data. It processes a dataset of bacterial promoters, extracts biologically relevant features, trains a classifier, and evaluates its performance.

## Features
- **Data Preprocessing:** Cleans and prepares promoter data from a CSV file.
- **Feature Extraction:** Computes sequence-based features (GC content, AT content, dinucleotide repeats, TATA box presence, Shannon entropy, k-mer counts, sequence length).
- **Model Training:** Uses a Random Forest classifier to distinguish strong vs. weak promoters.
- **Evaluation:** Outputs accuracy and classification report to a log file.

## Dataset
- **File:** `promoters.csv`
- **Description:** Contains annotated bacterial promoter sequences and metadata.
- **Key columns used:**
  - `6)pmSequence`: DNA sequence of the promoter
  - `15)confidenceLevel`: Label for promoter strength (S = strong, W = weak)

## Main Scripts
- `main.py`: Runs the full pipeline (preprocessing, feature extraction, training, evaluation).
- `preprocessing_and_features.py`: Functions for data cleaning and feature extraction.
- `model_train_test.py`: Model training and test split logic.
- `evaluation.py`: Model evaluation and logging.

## Usage
1. **Install dependencies** (requires Python 3.7+):
   ```bash
   pip install pandas numpy scikit-learn scipy
   ```
2. **Run the pipeline:**
   ```bash
   python main.py
   ```
   Results will be saved to `results_log.log`.

## Output
- **results_log.log:** Contains evaluation metrics and sample feature values.

## Customization
- You can adjust the k-mer size in `main.py` by changing the `k` parameter in `extract_all_features`.
- The model and features can be extended for other sequence classification tasks.
