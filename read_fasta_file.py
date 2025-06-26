import sys

def reader(fasta):
    try:
        with open(fasta, 'r') as f:
            lines = f.readlines()
            header = lines[0].strip()
            sequence = ''
            for line in lines[1:]:
                sequence += line.strip()
        
        print("The FASTA header is:", header)
        print("The FASTA sequence is:", sequence)
        
    except FileNotFoundError:
        print("File not found in directory")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Invalid argument. Usage: python script.py <filename>")
    fasta_file = sys.argv[1]
    reader(fasta_file)
