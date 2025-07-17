import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 1: Parse GFF and extract exons
def parse_exons(gff_file):
    print("\n[1] Parsing GFF for Exon Coordinates...")
    chrom_map = {
        'NC_004354.4': '2L', 'NC_004353.4': '2R', 'NC_004351.4': '3L',
        'NC_004350.4': '3R', 'NC_004352.4': 'X', 'NC_004355.4': '4',
        'NC_004356.4': 'Y', 'NC_024511.2': 'X', 'NC_024512.1': 'Y'
    }
    exons = []
    with open(gff_file) as f:
        for line in f:
            if line.startswith('#') or '\texon\t' not in line:
                continue
            parts = line.strip().split('\t')
            chrom = chrom_map.get(parts[0], parts[0])
            start, end = int(parts[3]), int(parts[4])
            attributes = parts[8]
            gene_id = None
            for tag in attributes.split(';'):
                if tag.startswith("Parent="):
                    gene_id = tag.split('=')[1]
                    break
            if gene_id:
                exons.append([chrom, start, end, gene_id])
    df = pd.DataFrame(exons, columns=['chrom', 'start', 'end', 'gene_id'])
    print(f"Parsed {len(df)} exons.")
    print(df.head(), "\n")
    return df

# Step 2: Parse SNPs from VCF
def parse_snps(vcf_file):
    print("[2] Parsing VCF for SNPs...")
    snps = []
    with open(vcf_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            snps.append([parts[0], int(parts[1])])
    df = pd.DataFrame(snps, columns=['chrom', 'pos'])
    print(f"Parsed {len(df)} SNPs.")
    print(df.head(), "\n")
    return df

# Step 3: Map SNPs to Exons
def map_snps_to_exons(exons, snps):
    print("[3] Mapping SNPs to Exons...")
    counts = {}
    for chrom in exons['chrom'].unique():
        snps_chr = snps[snps['chrom'] == chrom]
        exons_chr = exons[exons['chrom'] == chrom]
        for _, row in exons_chr.iterrows():
            gene = row['gene_id']
            matches = snps_chr[(snps_chr['pos'] >= row['start']) & (snps_chr['pos'] <= row['end'])]
            counts[gene] = counts.get(gene, 0) + len(matches)
    mapped = pd.DataFrame(counts.items(), columns=['gene_id', 'snp_count'])
    print(f"Mapped SNPs to {len(mapped)} genes.")
    print(mapped.head(), "\n")
    return mapped

# Normalize gene ID
def normalize_id(gid):
    if pd.isna(gid): return None
    gid = str(gid).replace('rna-', '').replace('gene-', '').replace('Dmel_', '')
    return gid.split('.')[0]

# Step 4: Build gene ID mapping
def build_gene_map(map_file):
    print("[4] Building Gene Mapping...")
    df = pd.read_csv(map_file, sep='\t')
    mapping = {}
    for _, row in df.iterrows():
        fbgn = str(row['Gene stable ID']).strip()
        for col in ['RefSeq mRNA ID', 'Transcript stable ID', 'FlyBase gene ID', 'Gene name']:
            val = str(row.get(col, '')).strip()
            if val:
                mapping[normalize_id(val)] = fbgn
    print(f"Mapping entries: {len(mapping)}\n")
    return mapping

# Step 5: Read gene expression
def read_expression(expr_file):
    print("[5] Reading Gene Expression File...")
    df = pd.read_csv(expr_file)
    df = df.rename(columns={'primary_FBid': 'gene_id'})
    df['expression'] = df.drop(columns=['gene_id', 'current_symbol'], errors='ignore').mean(axis=1)
    df = df[['gene_id', 'expression']]
    print(f"Parsed {len(df)} expression records.")
    print(df.head(), "\n")
    return df

# Final: Complete pipeline
def run(gff, vcf, expr_file, map_file, output_csv="snp_density_expression_merged.csv"):
    exons = parse_exons(gff)
    snps = parse_snps(vcf)
    snp_df = map_snps_to_exons(exons, snps)

    # Calculate exon lengths
    exon_lengths = exons.assign(length=exons.end - exons.start + 1).groupby('gene_id')['length'].sum()
    snp_df['exon_length'] = snp_df['gene_id'].map(exon_lengths)
    snp_df['snp_density'] = snp_df['snp_count'] / snp_df['exon_length']

    # Expression data
    expr_df = read_expression(expr_file)
    snp_df['gene_norm'] = snp_df['gene_id'].apply(normalize_id)
    expr_df['gene_norm'] = expr_df['gene_id'].apply(normalize_id)

    # ID mapping
    if os.path.exists(map_file):
        gene_map = build_gene_map(map_file)
        snp_df['gene_map'] = snp_df['gene_norm'].map(gene_map).fillna(snp_df['gene_norm'])
        expr_df['gene_map'] = expr_df['gene_norm'].map(gene_map).fillna(expr_df['gene_norm'])
    else:
        snp_df['gene_map'] = snp_df['gene_norm']
        expr_df['gene_map'] = expr_df['gene_norm']

    # Merge data
    merged = pd.merge(snp_df, expr_df, on='gene_map')
    merged = merged.dropna(subset=['snp_density', 'expression'])
    print(f"Merged dataset contains {len(merged)} genes.\n")
    print(merged.head(), "\n")

    # Save CSV
    merged.to_csv(output_csv, index=False)
    print(f"Saved merged data to: {output_csv}\n")

    # Correlation and plot
    merged['log_expr'] = np.log2(merged['expression'] + 1)
    merged['log_density'] = np.log2(merged['snp_density'] + 1)
    r = merged['log_expr'].corr(merged['log_density'])
    print(f"[6] Pearson Correlation (log2): r = {r:.4f}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged, x='log_expr', y='log_density', alpha=0.5)
    plt.title(f"SNP Density vs Expression (Log2)\nr = {r:.2f}")
    plt.xlabel("Log2(Gene Expression + 1)")
    plt.ylabel("Log2(SNP Density + 1)")
    plt.tight_layout()
    plt.savefig("snp_vs_expression_log.png", dpi=300)
    plt.show()
    # Expression Mean vs Variance
    expr_grouped = expr_df.groupby("gene_map")["expression"].apply(list)
    mean_expr = expr_grouped.apply(lambda x: np.log2(np.mean(x) + 1))
    var_expr = expr_grouped.apply(lambda x: np.log2(np.var(x) + 1))

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=mean_expr, y=var_expr, alpha=0.5)
    plt.title("Gene Expression: Mean vs Variance (Log2 Normalized)")
    plt.xlabel("Mean Expression (log2)")
    plt.ylabel("Variance (log2)")
    plt.tight_layout()
    plt.savefig("expression_mean_variance.png", dpi=300)
    plt.show()

    # 8. Create figures to visualize the relationship between SNP density & gene expression
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=merged, x='expression', y='snp_density')
    plt.xlabel('Gene Expression Level')
    plt.ylabel('SNP Density (per exon)')
    plt.title('Scatter Plot: SNP Density vs Gene Expression')
    plt.tight_layout()
    plt.savefig('snp_density_vs_expression_scatter.png')
    plt.close()

    plt.figure(figsize=(8,6))
    sns.boxplot(x=pd.qcut(merged['expression'], q=4, labels=["Q1","Q2","Q3","Q4"]), y=merged['snp_density'])
    plt.xlabel('Gene Expression Quartile')
    plt.ylabel('SNP Density (per exon)')
    plt.title('Boxplot: SNP Density by Gene Expression Quartile')
    plt.tight_layout()
    plt.savefig('snp_density_by_expression_quartile_boxplot.png')
    plt.close()

    print("\nAnalysis complete.")
    print(" Plots saved as:")
    print("   • snp_vs_expression_log.png")
    print("   • expression_mean_variance.png")
    print("   • snp_density_vs_expression_scatter.png")
    print("   • snp_density_by_expression_quartile_boxplot.png")

# --- Execute the pipeline ---
if __name__ == "__main__":
    run(
        gff="drosophila_melanogaster.gff",
        vcf="snps.vcf",
        expr_file="GSE263568_non_allele_pseudotime_normalized_reads.csv",
        map_file="mart_export.txt"
    )