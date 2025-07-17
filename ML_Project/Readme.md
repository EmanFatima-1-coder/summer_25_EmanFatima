# ML_Project_1
## Predicting Drug-like Ligands Using Molecular Descriptors and Atom Encodings

This project develops a machine learning model to classify ligands as drug-like or non-drug-like using molecular descriptors and atom encodings.

### Workflow Overview:
1. **Data Collection**: Downloaded drug-like and non-drug-like compounds in `.sdf` format and stored them in separate folders.
2. **Labeling**: Labeled each compound based on its class.
3. **Feature Extraction**: Used RDKit to extract atom-level encodings and molecular descriptors.
4. **Dataset Preparation**: Compiled features into a structured dataset suitable for machine learning.
5. **Model Building**: Trained ML models (e.g., Random Forest, Logistic Regression) to classify ligands.
6. **Evaluation**: Assessed models using accuracy, precision, recall, F1-score, and ROC-AUC.
7. **Visualization**: Plotted performance metrics for interpretability.

###  Technologies Used:
- Python
- RDKit
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

###  Key Learning Objectives:
- Understand ligands and the concept of drug-likeness.
- Explore the role of molecular descriptors in predicting bioactivity.
- Gain hands-on experience with cheminformatics and ML pipelines.
- Interpret classification metrics in the context of model evaluation.

