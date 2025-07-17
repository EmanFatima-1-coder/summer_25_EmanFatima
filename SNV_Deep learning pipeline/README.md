# SNV Deep Learning Pipeline

This project provides a deep learning pipeline for analyzing and classifying Single Nucleotide Variants (SNVs) using an Artificial Neural Network (ANN) implemented in PyTorch. The pipeline includes data loading, feature engineering, model training with early stopping, evaluation, and visualization of results.

## Project Structure

- `annotated_data.py` — Script for handling and processing annotated SNV data.
- `annotated_snv_data.csv` — Example dataset containing annotated SNV data.
- `data_loader.py` — Utilities for loading and preprocessing data.
- `features.py` — Feature engineering and extraction functions.
- `train_model.py` — Defines the ANN model, training loop (with early stopping), and plotting utilities for loss, confusion matrix, and ROC curve.
- `evaluate_model.py` — Script for evaluating the trained model and generating performance metrics.
- `main.py` — Main entry point to run the pipeline end-to-end.
- `roc_curve.png`, `training_loss.png` — Example output visualizations.
- `SNV_Deep_Learning_Report.docx` — Project report/documentation.

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- scikit-learn
- matplotlib

Install dependencies with:

```bash
pip install torch numpy scikit-learn matplotlib
```

## Usage

1. **Prepare Data:**
   - Place your annotated SNV data in `annotated_snv_data.csv` or update the data path in the scripts.

2. **Feature Engineering:**
   - Use `features.py` to extract and engineer features from the raw data.

3. **Train Model:**
   - Run `train_model.py` to train the ANN model. Training loss and model checkpoints will be saved.

4. **Evaluate Model:**
   - Use `evaluate_model.py` to assess model performance. This will generate confusion matrix and ROC curve visualizations.

5. **Run Full Pipeline:**
   - Execute `main.py` to run the entire workflow from data loading to evaluation.

## Output

- Training loss plot: `visualizations/training_loss.png`
- Confusion matrix: `visualizations/confusion_matrix.png`
- ROC curve: `visualizations/roc_curve.png`

## Notes
- The model uses early stopping to prevent overfitting.
- All visualizations are saved in the `visualizations/` directory (created automatically).
- You can adjust model parameters and training settings in `train_model.py`.

 