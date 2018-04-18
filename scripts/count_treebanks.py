import os.path
ID,FORM,LEMMA,CPOS,POS,FEAT,HEAD,DEPREL,DEPS,MISC=range(10)


def read_conllu(f):
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

def count_words(fname):
    words=0
    if not os.path.isfile(fname):
        return 0
    with open(fname, "rt") as f:
        for comm, sent in read_conllu(f):
            words+=len(sent)
    return words

with open("/home/jmnybl/git_checkout/universal-lemmatizer/morpho_transducers/iso_names.txt", "rt") as f:
    treebanks=[t.split(" ")[0] for t in f]

path_v2="/home/jmnybl/UD-2.0/reproducible_training/udpipe-ud-2.0-170801-reproducible_training/ud-2.0/"
path_st="/home/jmnybl/conll17-data/ud-treebanks-conll2017/all/"

for t in treebanks:
    for s in ["train", "dev"]:#, "test"]:
        fname=path_v2+"{lang}/{lang}-ud-{dataset}.conllu".format(lang=t,dataset=s)
        c_v2=count_words(fname)
        fname=path_st+"{lang}-ud-{dataset}.conllu".format(lang=t,dataset=s)
        c_st=count_words(fname)
        print("{lang}-ud-{dataset}.conllu".format(lang=t,dataset=s)," RESPLITTED:",c_v2, ", ST:",c_st, c_v2==c_st)
    
