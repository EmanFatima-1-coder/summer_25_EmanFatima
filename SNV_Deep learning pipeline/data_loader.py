import pandas as pd

vcf_file = "clinvar.vcf"

records = []

with open(vcf_file, "r") as file:
    for line in file:
        # Skip header lines
        if line.startswith("#"):
            continue
        
        cols = line.strip().split('\t')
        chrom = cols[0]
        pos = int(cols[1])
        ref = cols[3]
        alt = cols[4]
        info = cols[7]

        # Find CLNSIG from INFO column
        clnsig = None
        for entry in info.split(";"):
            if entry.startswith("CLNSIG="):
                clnsig = entry.split("=")[1].split('|')[0]  # Take first if multiple
                break
        
        if not clnsig:
            continue
        
        # Define labels
        if "Pathogenic" in clnsig or "Likely_pathogenic" in clnsig:
            label = 1
        elif "Benign" in clnsig or "Likely_benign" in clnsig:
            label = 0
        else:
            continue  # Skip Uncertain/conflicting

        records.append({
            "CHROM": chrom,
            "POS": pos,
            "REF": ref,
            "ALT": alt,
            "CLNSIG": clnsig,
            "Label": label
        })

# Create DataFrame
df = pd.DataFrame(records)

# Save to CSV
df.to_csv("filtered_snv_data.csv", index=False)
print(f"Saved {len(df)} SNVs to filtered_snv_data.csv")
