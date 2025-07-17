# DNA Sequence Analysis Project

This project analyzes DNA sequences from a FASTA file, providing insights such as GC content, EcoRI restriction site counts, transcription/translation products, and sequence length. It generates summary plots and a CSV file with the results.

## Features
- **Input Validation:** Reads and validates DNA sequences from a FASTA file.
- **GC Content Calculation:** Computes the GC content for each sequence.
- **Restriction Site Analysis:** Counts EcoRI restriction sites in each sequence.
- **Transcription & Translation:** Provides the first 50 bases of RNA and protein translation for each sequence.
- **Visualization:**
  - Bar plot of GC content per sequence (`gc_content_bar.png`)
  - Histogram of sequence lengths (`length_histogram.png`)
  - Interactive scatter plot of GC content vs. EcoRI sites (`restriction_scatter.html`)
- **Tabular Output:** Saves all results to `sequence_analysis.csv`.

## Requirements
- Python 3.7+
- Packages:
  - biopython
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - plotly

Install dependencies with:
```bash
pip install biopython numpy pandas matplotlib seaborn plotly
```

## Usage
1. Place your FASTA file (e.g., `Saccharomyces_cerevisiae.fasta`) in the project directory.
2. Run the analysis script:
   ```bash
   python Data_Analysis.py
   ```
3. Outputs will be generated in the same directory:
   - `sequence_analysis.csv`
   - `gc_content_bar.png`
   - `length_histogram.png`
   - `restriction_scatter.html`

## Notes
- The script expects the FASTA file to be named `Saccharomyces_cerevisiae.fasta` by default. You can change the filename in the script if needed.
- Sequences with invalid characters are skipped with a warning.

## Example Output
- **CSV:** Tabular summary of all analyzed sequences.
- **Plots:** Visual summaries for quick insights.

---
