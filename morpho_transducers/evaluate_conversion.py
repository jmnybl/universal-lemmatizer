import sys

ID,FORM,LEMMA,UPOS,POS,FEAT,HEAD,DEPREL,DEPS,MISC=range(10)

def conllu_reader(f):
    sent=[]
    comment=[]
    for line in f:
        line=line.strip()
        if not line: # new sentence
            if sent:
                yield comment,sent
            comment=[]
            sent=[]
        elif line.startswith("#"):
            comment.append(line)
        else: #normal line
            sent.append(line.split("\t"))
    else:
        if sent:
            yield comment, sent
    

def read_treebank(args):
    words={}
    for comm, sent in conllu_reader(open(args.treebank,"rt",encoding="utf-8")):
        for token in sent:
            if token[FORM] not in words:
                words[token[FORM]]=[]
            if (token[LEMMA],token[UPOS],token[FEAT]) not in words[token[FORM]]:
                words[token[FORM]].append((token[LEMMA],token[UPOS],token[FEAT]))
    return words

def transducer_reader():
    readings=[]
    for line in sys.stdin:
        line=line.strip()
        if not line:
            if readings:
                yield readings
                readings=[]
            continue
        if line.split("\t") not in readings:
            readings.append(line.split("\t"))

def calculate_tags(transducer,treebank):

    transducer_tags=[transducer[2]]+(transducer[3].split("|") if transducer[3] != "_" else [])
    treebank_tags=[treebank[1]]+(treebank[2].split("|") if treebank[2] != "_" else [])
    print(transducer_tags,treebank_tags)
    return len(set(transducer_tags)&set(treebank_tags)),len(set(transducer_tags)-set(treebank_tags)),len(set(treebank_tags)-set(transducer_tags)),1 if transducer[1]==treebank[0] else 0

def evaluate(args):

    treebank_words=read_treebank(args)

    true_positives,false_positives,false_negatives=0,0,0
    lemmas=0
    counter=0
    for readings in transducer_reader():
        word=readings[0][0]
        if word in treebank_words and len(readings)==1 and "*" not in readings[0][1] and len(treebank_words[word])==1: # TODO do not skip, do something wiser!
            print(word,readings,treebank_words.get(word,"_"))
            tp,fp,fn,l=calculate_tags(readings[0],treebank_words[word][0])
            true_positives,false_positives,false_negatives,lemmas=true_positives+tp,false_positives+fp,false_negatives+fn,lemmas+l
            counter+=1


    print("Tag precision:",true_positives/(true_positives+false_positives) if (true_positives!=0 or false_positives!=0) else 0)
    print("Tag recall:",true_positives/(true_positives+false_negatives) if (true_positives!=0 or false_positives!=0) else 0)
    print("Lemma accuracy:",lemmas/counter if counter>0 else 0)




if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    
    g.add_argument('-t', '--treebank', type=str, help='Input treebank file name')
    g.add_argument('--max_words', type=int, default=10000, help='How many words to read from converted transducer output')
    
    args = parser.parse_args()

    evaluate(args)
