import sys
import os

def read_fasta(file_path):
    """Read a FASTA file and return a dictionary of header:sequence pairs"""
    fasta_dict = {}
    try:
        with open(file_path, 'r') as file:
            header = None
            sequence_parts = []
            for line in file:
                line = line.strip()
                if not line:
                    continue  # skip empty lines
                if line.startswith(">"):
                    if header:
                        fasta_dict[header] = ''.join(sequence_parts)
                    header = line
                    sequence_parts = []
                elif header:
                    if not all(c in "ATGCatgcNn" for c in line):  # basic FASTA format check
                        print(f"Warning: Invalid characters in sequence: {line}")
                        continue
                    sequence_parts.append(line)
                else:
                    print(f"Warning: Sequence found before header: {line}")
            if header:
                fasta_dict[header] = ''.join(sequence_parts)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    return fasta_dict

def filter_sequences(fasta_dict, min_length):
    """Filter sequences shorter than min_length"""
    return {header: seq for header, seq in fasta_dict.items() if len(seq) >= min_length}

def write_fasta(filtered_dict, output_file):
    """Write filtered sequences to a new FASTA file"""
    try:
        with open(output_file, 'w') as f:
            for header, sequence in filtered_dict.items():
                f.write(f"{header}\n")
                # Wrap sequence in lines of 60 characters
                for i in range(0, len(sequence), 60):
                    f.write(sequence[i:i+60] + '\n')
    except IOError as e:
        print(f"Error writing to output file: {e}")
        sys.exit(1)

def display_summary(total, passed):
    print("\nSummary:")
    print(f"Total sequences read: {total}")
    print(f"Sequences passed length filter: {passed}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_fasta> <output_fasta>")
        sys.exit(1)

    input_fasta = sys.argv[1]
    output_fasta = sys.argv[2]

    # Prompt for minimum sequence length
    try:
        min_len = int(input("Enter minimum sequence length to keep: "))
        if min_len < 0:
            raise ValueError("Length must be non-negative.")
    except ValueError as ve:
        print(f"Invalid input: {ve}")
        sys.exit(1)

    # Read and process
    sequences = read_fasta(input_fasta)
    total_sequences = len(sequences)

    filtered = filter_sequences(sequences, min_len)
    passed_sequences = len(filtered)

    write_fasta(filtered, output_fasta)
    display_summary(total_sequences, passed_sequences)

if __name__ == "__main__":
    main()
