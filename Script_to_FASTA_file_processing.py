import csv
# function to find gc content of a sequence
def gc_content(dna_seq):
    gc_count=dna_seq.count('G') + dna_seq.count('C')
    return (gc_count / len(dna_seq)) * 100

# function to check if sequence is valid or not
def is_validate(dna_seq):
    is_valid=all(base in 'ATGC' for base in dna_seq)
    return(is_valid)

# function to get unique nucleotides
def get_unique_nucleotides (dna_seq):
    unique_nucleotides = set()
    for seq in dna_seq.values():
        unique_nucleotides.update(seq)
    return unique_nucleotides

# function to read fasta file
def read_fasta_file(file_path):
    sequences={}
    with open(file_path, 'r') as file:
        current_id=""
        for line in file:
            line= line.strip()
            if line.startswith('>'):
                current_id = line[1:]
                sequences[current_id]=""
            else:
                sequences[current_id]+=line
    return sequences

# function to save output to csv file
def save_to_csv(sequences,output_csv):
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id","length", "gc_content", "valid_dna"])
        for id, seq in sequences.items():
            length= len(seq)
            gc_content_val = gc_content(seq)
            is_valid = is_validate(seq)
            writer.writerow([id, len(seq), gc_content_val, is_valid])
       

if __name__=="__main__":
    fasta_file="input.fasta"
    output_csv="output.csv"
    sequences = read_fasta_file(fasta_file)
    unique_nucs = get_unique_nucleotides(sequences)
    print("Unique nucleotides across all sequences:", unique_nucs)
    save_to_csv(sequences, output_csv)
    print(f"Analysis results saved to {output_csv}")