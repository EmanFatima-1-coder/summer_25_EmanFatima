import sys
from Bio import pairwise2
from Bio.Seq import Seq

def alignment(seq1, seq2, match=2, mismatch=-2, gap_open=-2, gap_extended=-2):
    alignment = pairwise2.align.globalms(seq1, seq2, match, mismatch, gap_open, gap_extended)
    best_alignment = alignment[0]

    print("Alignment Score:", best_alignment.score)
    print("Aligned Sequence 1:", best_alignment.seqA)
    print("Aligned Sequence 2:", best_alignment.seqB)
    print("Start of Alignment:", best_alignment.start)
    print("End of Alignment:", best_alignment.end)
    return best_alignment

def similarity(alignment):
    aligned1 = alignment.seqA
    aligned2 = alignment.seqB

    matches = 0
    aligned_length = len(aligned1)

    for i in range(aligned_length):
        if aligned1[i] == aligned2[i] and aligned1[i] != '-':
            matches += 1

    similarity = (matches / aligned_length) * 100 if aligned_length > 0 else 0
    print("Similarity:", similarity, "%")
    return similarity

def calculate_gap_percentage(alignment):
    aligned1 = alignment.seqA
    aligned2 = alignment.seqB

    gaps_seq1 = aligned1.count('-')
    gaps_seq2 = aligned2.count('-')
    total_gaps = gaps_seq1 + gaps_seq2

    alignment_length = len(aligned1)

    gap_percentage = (total_gaps / (alignment_length * 2)) * 100 if alignment_length > 0 else 0

    print("Gap Percentage:", gap_percentage, "%")
    return gap_percentage

def find_conserved(alignment, threshold=20):
    aligned1 = alignment.seqA
    aligned2 = alignment.seqB

    conserved_regions = []
    start = None
    length = 0

    for i in range(len(aligned1)):
        if aligned1[i] == aligned2[i] and aligned1[i] != '-':
            if start is None:
                start = i
            length += 1
        else:
            if length >= threshold:
                conserved_regions.append((start, i - 1, aligned1[start:i]))
            start = None
            length = 0

    if length >= threshold:
        conserved_regions.append((start, len(aligned1) - 1, aligned1[start:]))

    return conserved_regions

if __name__ == "__main__":
    if len(sys.argv) != 3:
        seq1 = "ATGCGTGCGATGCGTGtTGCAGTGACTGACTGACCCCGGTAA"
        seq2 = "AGCCGTGCCATGCGTGACGTGACTTTCSTACTCGTAGCTG"
        print("No command-line arguments provided. Using default sequences.")
    else:
        seq1 = sys.argv[1]
        seq2 = sys.argv[2]

    align_result = alignment(seq1, seq2)
    similarity(align_result)
    calculate_gap_percentage(align_result)
    align_conserved = pairwise2.align.globalxx(seq1, seq2)[0]
    conserved = find_conserved(align_conserved, threshold=5) 

    for region in conserved:
        print(f"Start: {region[0]}, End: {region[1]}, Sequence: {region[2]}")
