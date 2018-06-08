import sys
import os
import numpy as np
from random import shuffle
from prepare_data import read_conllu,ID,FORM,LEMMA,UPOS,XPOS,FEAT,HEAD,DEPREL,DEPS,MISC

def read_character_probabilities(vocab):

    with open(vocab,"rt",encoding="utf-8") as f:
        characters=[]
        counts=[]
        for line in f:
            line=line.lstrip().strip("\n")
            if not line:
                continue
            try:
                count,char=line.split(" ",1)
                characters.append(char)
                counts.append(int(count))
            except:
                print("Error with line",line,file=sys.stderr)
                continue
        if " " not in characters: # add whitespace if it's not there
            print("Adding whitespace into vocabulary", file=sys.stderr)
            characters.append(" ")
            counts.append(100)
        counts=[c/sum(counts) for c in counts]
        print("Char probs:",counts[:10], file=sys.stderr)

    return characters, counts


def create_character_probabilities(conllu_file):

    characters={}
    for comm, sent in read_conllu(open(conllu_file, "rt")):
        for token in sent:
            for char in token[FORM]:
                if char not in characters:
                    characters[char]=0
                characters[char]+=1
    sorted_chars = sorted(characters.items(), key=lambda x: x[1])
    chars=[c for (c,_) in sorted_chars]
    counts=[c for (_,c) in sorted_chars]
    counts=[c/sum(counts) for c in counts]
    print("Char probs:",counts[:10], file=sys.stderr)
    return chars, counts

def create_data(vocabulary, example_count, extra_tag):

    if vocabulary.endswith(".conllu"):
        characters, counts = create_character_probabilities(vocabulary)
    else:
        characters, counts = read_character_probabilities(vocabulary)

    #if not os.path.exists(os.path.dirname(args.output)):
    #    os.makedirs(os.path.dirname(args.output))

    #f_inp=open(args.output+".input","wt",encoding="utf-8")
    #f_out=open(args.output+".output","wt",encoding="utf-8")

    counter=0
    selector=0
    data=[]
    while counter<example_count:

        # create random strings based on character distribution
        # --> but guarantee that each character is sampled at least once
        #     because we want to keep complete vocabulary
        if selector==len(characters):
            selector=0
        chars=[characters[selector]]
        
        chars+=list(np.random.choice(characters,np.random.randint(2,12),replace=True,p=counts)) # takes character probabilities
        shuffle(chars)    
        lemma=" ".join(c if c!=" " else "$@@$" for c in chars) # replace whitespace with $@@$
        wordform=" ".join(c if c!=" " else "$@@$" for c in chars)
            
        tags=[]
        if extra_tag!="":
            tags.append(extra_tag)
        if tags:
            tags=" ".join(tags)
            wordform=" ".join([wordform, tags])
        data.append((wordform, lemma))
        counter+=1
        selector+=1
    print("Done, produced",counter,"examples.",file=sys.stderr)
    return data


def main(args):

    data=create_data(args.vocabulary, args.count, args.extra_tag)

    # print to files
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    f_inp=open(args.output+".input","wt",encoding="utf-8")
    f_out=open(args.output+".output","wt",encoding="utf-8")

    for (word, lemma) in data:
        print(word,file=f_inp)
        print(lemma,file=f_out)

    f_inp.close()
    f_out.close()

if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    
    g.add_argument('-v', '--vocabulary', type=str, help='Name of the vocabulary (counted characters)')
    g.add_argument('-o', '--output', type=str, help='Output file name, will create file extentions .input and .output')
    g.add_argument('--count', type=int, default=200000, help='How many random string to generate')
    g.add_argument('--extra_tag', type=str, default="", help='extra tag, for example mark autoencoding training examples')
    
    args = parser.parse_args()

    main(args)
