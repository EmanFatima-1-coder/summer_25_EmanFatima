import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import sys

def protein_feature_analysis(fasta_file):
    try:
        all_features=[]
        for record in SeqIO.parse(fasta_file,'fasta'):
            sequence= str(record.seq).replace("_","").replace("*","")
            protein= ProteinAnalysis(str(sequence))

            features={
                # "id":record.id,
                "mw":protein.molecular_weight(),
                "isoelectric_point":protein.isoelectric_point(),
                "aromaticity":protein.aromaticity(),
                "instability_index":protein.instability_index(),
                # "gravy":protein.gravy(),
            }
            all_features.append(features)
        df= pd.DataFrame(all_features)
        print("all features:",df)
        return df


    except FileNotFoundError:
        print(f"Error: The file {fasta_file} was not found")
        sys.exit(1)

def synthetic_labels(df):
    try:
        labels=['cytoplasm', 'nucleus', 'mitochondria']
        df['labels']=np.random.choice(labels, size=len(df))
        print(df)
        return df
    except Exception as e:
        print(f"no dataset found{e}")
        sys.exit(1)