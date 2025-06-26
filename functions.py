import sys

def seq_concat(a, b):
    seq1 = a
    seq2 = b
    concat = seq1 + seq2
    print('concatenation is', concat)

def gc_content(dna):
    length = len(dna)
    gc_count = dna.count('G') + dna.count('C')
    gc_ratio = gc_count / length
    print("gc content:", gc_ratio)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit('usage: python file_name arg_1 arg_2')
    
    seq_1 = sys.argv[1]
    seq_2 = sys.argv[2]

    seq_concat(seq_1, seq_2)
    print("DNA sequence:", seq_1)
    gc_content(seq_1)
