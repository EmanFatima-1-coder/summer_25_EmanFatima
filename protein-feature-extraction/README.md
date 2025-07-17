# Protein Feature Extraction and Modeling

This project provides tools for extracting features from protein sequences and building predictive models using those features. It is designed to work with protein data in FASTA format and includes scripts for feature extraction, model training, and prediction.

## Project Structure

- `extract_features.py`: Script for extracting features from protein sequences in a FASTA file.
- `model.py`: Contains code for building and using machine learning models on the extracted features.
- `main.py`: Main entry point for running the feature extraction and modeling pipeline.
- `proteins.fasta`: Example input file containing protein sequences in FASTA format.
- `output.log`: Log file for output and results.

## Usage

1. **Extract Features:**
   Run `extract_features.py` to extract features from your protein FASTA file.
   ```bash
   python extract_features.py proteins.fasta
   ```

2. **Train Model:**
   Use `main.py` to run the full pipeline, including feature extraction and model training.
   ```bash
   python main.py
   ```

3. **Model Prediction:**
   Use `model.py` to load a trained model and make predictions on new data.

## Requirements

- Python 3.7+
- Common scientific libraries: numpy, pandas, scikit-learn

Install dependencies with:
```bash
pip install -r requirements.txt
```

## License

This project is provided under the MIT License. 