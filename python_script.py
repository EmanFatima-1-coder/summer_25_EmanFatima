DNA_SEQ="TCCAACCGTCCACAGTGAGCCACGTCGCCCAACAAATCTCACACAACAGATCCAGATCCGACAC"
print(DNA_SEQ)
dna_int=4
dna_float=5.006
dna="ATGC"
length=4
is_dna=True
dna=5
rna=6
add=dna+rna
print("concatenation==",add)
dna_1="ATGC"
dna_2="ATGGC"
print(type(dna))
print(type(length))
print(type(is_dna))
dna="ATGCCGATTTAACGC"
threshold=10
gc_content=dna.count('G') + dna.count('C')

if gc_content==threshold:
    print("GC content is equals to threshold")
elif gc_content > threshold:
    print("GC content is not equals to threshold")
elif gc_content < threshold:
    print("GC content is less than threshold")
else:
    print(gc_content)
    






