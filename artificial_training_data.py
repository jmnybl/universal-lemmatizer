import sys
import os
import numpy as np
from random import shuffle

def create_data(args):

    with open(args.vocabulary,"rt",encoding="utf-8") as f:
        characters=[]
        counts=[]
        for line in f:
            line=line.lstrip().strip("\n")
            if not line:
                continue
            try:
                count,char=line.split()
                characters.append(char)
                counts.append(int(count))
            except:
                print("Error with line",line,file=sys.stderr)
                continue
        counts=[c/sum(counts) for c in counts]
        print(counts[:10])

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    f_inp=open(args.output+".input","wt",encoding="utf-8")
    f_out=open(args.output+".output","wt",encoding="utf-8")

    counter=0
    selector=0
    while counter<args.count:

        # create random strings based on character distribution
        # --> but guarantee that each character is sampled at least once
        # --> because we want to keep complete vocabulary
        if selector==len(characters):
            selector=0
        chars=[characters[selector]]
        
        chars+=list(np.random.choice(characters,np.random.randint(2,12),replace=True,p=counts)) # can take probabilities
        shuffle(chars)    
        lemma=" ".join(c for c in "".join(chars).replace(" ","$@@$")) # replace whitespace with $@@$
        wordform=" ".join(c for c in "".join(chars).replace(" ","$@@$"))
            
        tags=[]
        if args.extra_tag!="":
            tags.append(args.extra_tag)
           
        tags=" ".join(tags)
        print(wordform,tags,file=f_inp)
        print(lemma,file=f_out)
        counter+=1
        selector+=1
    print("Done, files have",counter,"examples.",file=sys.stderr)



if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    
    g.add_argument('-v', '--vocabulary', type=str, help='Name of the vocabulary (counted characters)')
    g.add_argument('-o', '--output', type=str, help='Output file name, will create file extentions .input and .output')
    g.add_argument('--count', type=int, default=200000, help='How many random string to generate')
    g.add_argument('--extra_tag', type=str, default="", help='extra tag, for example mark autoencoding training examples')
    
    args = parser.parse_args()

    create_data(args)
