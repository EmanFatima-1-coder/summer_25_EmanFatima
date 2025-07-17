import pandas as pd
import requests
import time

data = pd.read_csv("filtered_snv_data.csv")

clean = []
for _, row in data.iterrows():
    if len(row['REF']) == 1 and len(row['ALT']) == 1 and ',' not in row['REF'] and ',' not in row['ALT']:
        clean.append(row)

df = pd.DataFrame(clean).sample(n=5000, random_state=42).reset_index(drop=True)

variants = []
for _, row in df.iterrows():
    variants.append(f"{row['CHROM']} {int(row['POS'])} . {row['REF']} {row['ALT']}")
df['variant'] = variants

def annotate(batch):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    r = requests.post("https://rest.ensembl.org/vep/human/region", headers=headers, json={"variants": batch})
    return r.json() if r.ok else []

result = []
for i in range(0, len(variants), 200):
    batch = variants[i:i+200]
    data = annotate(batch)
    for item in data:
        t = item.get("transcript_consequences", [{}])[0]
        result.append({
            "variant": item.get("input", ""),
            "gene": t.get("gene_symbol", ""),
            "consequence": item.get("most_severe_consequence", ""),
            "sift": t.get("sift_prediction", ""),
            "polyphen": t.get("polyphen_prediction", ""),
            "impact": t.get("impact", "")
        })
    time.sleep(1.5)

ann = pd.DataFrame(result)
final = df.merge(ann, on="variant", how="left")
final.to_csv("annotated_snv_data.csv", index=False)

print("Done")
