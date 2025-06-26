import sys

def parse(file):
    with open(file, 'r') as g:
        for line in g:
            if line.startswith("#") or line.strip() == "":
                continue  # skip comment or empty lines
            col = line.strip()
            record = col.split('\t')

            if len(record) < 6:
                continue  # skip malformed lines

            id = record[0]
            type = record[2]
            start_coord = record[3]
            end_coord = record[4]
            score = record[5]

            print('The ID is:', id)
            print('The type of region is:', type)
            print('The starting position is:', start_coord)
            print('The ending position is:', end_coord)
            print('The score is:', score)
            print('\n')

def write_gff(output_file):
    entries = [
        "chr1\tRefSeq\tgene\t1000\t5000\t.\t+\t.\tID=gene0001;Name=GeneA",
        "chr1\tRefSeq\tmRNA\t1000\t5000\t.\t+\t.\tID=mrna0001;Parent=gene0001;Name=GeneA-001",
        "chr1\tRefSeq\texon\t1000\t1500\t.\t+\t.\tID=exon0001;Parent=mrna0001",
    ]
    with open(output_file, 'w') as f:
        f.write("##gff-version 3\n")
        for entry in entries:
            f.write(entry + "\n")
    print(f"GFF3 file written to: {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Usage: python filename.py genome.gff3')
    
    gff = sys.argv[1]
    parse(gff)
    write_gff("output.gff3")
