import sys
dna=input("Enter your DNA sequence:")
print ("Your DNA sequence is:",dna)
print("Length of DNA sequence is:",len(dna))
GC_count=dna.count('G') + dna.count('C')
GC_content = GC_count / len(dna)
print (f"GC Content: {GC_content:.2f}")
Validate = all(base in 'ATGC' for base in dna)
print("Valid DNA:", Validate)
count = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
for base in dna:
   if base in count:
     count[base] += 1
print("Counts occurence for each nucleotide is:", count)
if Validate and GC_content>0.4:
  print("High GC content:")
else:
     print("low GC content:")
index = len(dna)- 1
reversed_dna = ""
while index >= 0:
 reversed_dna += dna[index]
 index-= 1
print("Original DNA:", dna)
print("Reversed DNA:", reversed_dna)

    
   

