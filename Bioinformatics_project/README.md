# Comparative Analysis of SNP Distribution and Gene Expression Across Diverse Organisms

## Project Overview
This project analyzes the distribution of single nucleotide polymorphisms (SNPs) in exonic regions and correlates them with gene expression levels in a chosen organism (e.g., Drosophila melanogaster). The analysis provides insights into evolutionary pressures such as purifying selection.

## Objectives
- Parse genome annotation (GFF) to extract exonic coordinates for each gene.
- Parse SNP data (VCF) and map SNPs to exons.
- Calculate SNP density per gene.
- Correlate SNP density with gene expression levels.
- Create visualizations to illustrate the relationship between SNP density and gene expression.
- Perform Pearson correlation analysis.
- Interpret results in an evolutionary context.

## Data Files
- `drosophila_melanogaster.gff`: Genome annotation file.
- `snps.vcf`: SNP data file.
- `GSE263568_non_allele_pseudotime_normalized_reads.csv`: Gene expression data.
- `snp_density_expression_merged.csv`: Output file with merged SNP density and expression data.

## How to Run
1. Ensure you have Python 3 and the required libraries installed:
   - pandas
   - biopython
   - matplotlib
   - seaborn
   - scipy
2. Run the main analysis script:
   ```bash
   python snp_expression_analysis.py
   ```
3. The script will generate output files and figures in the project directory.

## Output
- Figures visualizing the relationship between SNP density and gene expression.
- CSV file with merged SNP density and expression data.
- Report summarizing methodology, results, and biological interpretation.

## Visualization
The script generates figures (e.g., scatter plots, boxplots) to visualize the relationship between SNP density and gene expression. These figures are saved in the project directory and referenced in the report.

