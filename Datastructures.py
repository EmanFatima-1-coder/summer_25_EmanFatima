# Datastructures
# list
...
list=[]
tuple ={}
dictionary={}
#list

gene_list=['brac1','tp53','gc35']
gene_list.append("gene4")# to insert new gene
gene_list.insert(2,"gene4")# to insert on specific position
gene_list.remove("gene2")
removed=gene_list.pop(2)
print(removed)
print(gene_list.index("gene3"))
gene_list.sort()
num_list=[5,3,2,1]
num_list.sort()
num_list.reverse()
num_list.clear()
print("the list of gene is:",gene_list)
gene_tuple=["g1","g2","g3"]
print(gene_tuple)
print("number of list in sorted way:",num_list)
# tuple
gene_1=(100,200),(500,600)
for start,end in gene_1:
    print("the position og gene_1 is start from:,start, the ending in this position is:")
    ####
    gene_dict={"brac1":1,"tp53":2}
    print(type(gene_dict))
    for name,num in gene_dict():
        print("the name of gene is:",name,"and the number is")

    


