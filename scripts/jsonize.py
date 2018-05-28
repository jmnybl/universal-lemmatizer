import sys
import os
import json
import glob

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


def gather_names(data):

    # gather all full names (e.g. UD_Finnish-TDT) adn init some of the variables in dictionary
    files = glob.glob("/usr/share/ParseBank/ud-2.2-st-train-dev-data/ud-treebanks-v2.2/*/*README*")

    for f in files:
        _,name,_=f.strip().rsplit("/",2)
        data[name]={}
        data[name]["treebank_code"]="NULL"
        data[name]["train_size_sentences"]=0
        data[name]["train_size_tokens"]=0
        data[name]["dev_size_sentences"]=0
        data[name]["dev_size_tokens"]=0
        data[name]["baseline_udpipe_resplitted"]="NULL"
        data[name]["baseline_udpipe_lemma_dev"]="NULL"

    # gather treebank codes (e.g. fi_tdt) (note that some repos does not have training data and this will be empty!)
    files = glob.glob("/usr/share/ParseBank/ud-2.2-st-train-dev-data/ud-treebanks-v2.2/*/*-train.conllu")

    for f in files:
        _,name,_=f.strip().rsplit("/",2)
        _,filename=f.strip().rsplit("/",1)
        code,_=filename.split("-",1)
        data[name]["treebank_code"]=code


    return data

def count_sizes(files, prefix, data):

    for f in files:
        _,name,_=f.strip().rsplit("/",2)

        # count training data sizes
        with open(f, "rt") as c:
            sentences=0
            tokens=0
            for comm, sent in read_conllu(c):
                sentences+=1
                tokens+=len(sent)

            data[name]["{p}_size_sentences".format(p=prefix)]=sentences
            data[name]["{p}_size_tokens".format(p=prefix)]=tokens

    return data

def compare_udpipe_sizes(data):

    for key in data.keys():

        tb_code=data[key]["treebank_code"]
        if tb_code != "NULL":

            f="/usr/share/ParseBank/ud-2.2-st-train-dev-data/baseline-models-conll18-udv2.2/training/training_data/{c}/{c}-ud-train.conllu".format(c=tb_code)
            with open(f, "rt") as c:
                sentences=0
                tokens=0
                for comm, sent in read_conllu(c):
                    sentences+=1
                    tokens+=len(sent)

                if data[key]["train_size_sentences"]==sentences and data[key]["train_size_tokens"]==tokens:
                    data[key]["baseline_udpipe_resplitted"]=False
                else:
                    data[key]["baseline_udpipe_resplitted"]=True

    return data

def gather_basic_treebank_info():

    data=gather_names({})

    train_files=glob.glob("/usr/share/ParseBank/ud-2.2-st-train-dev-data/ud-treebanks-v2.2/*/*-train.conllu")
    devel_files=glob.glob("/usr/share/ParseBank/ud-2.2-st-train-dev-data/ud-treebanks-v2.2/*/*-dev.conllu")

    data=count_sizes(train_files, "train", data)
    data=count_sizes(devel_files, "dev", data)

    data=compare_udpipe_sizes(data)

    return data

def get_udpipe_baseline_numbers(data):

    with open("/usr/share/ParseBank/ud-2.2-st-train-dev-data/baseline-models-conll18-udv2.2/README.txt", "rt") as f:

        for line in f:
            cols=line.strip().split("|")
            if len(cols)!=15 or line.startswith("Treebank") or cols[1].strip()!="Gold tok":
                continue
            name="UD_"+cols[0].strip().replace(" ", "_")
            if name not in data:
                print("skipping", name, file=sys.stderr)
                continue
            lemma=cols[8].strip().replace("%","")
            if lemma != "-":
                data[name]["baseline_udpipe_lemma_dev"]=float(lemma)

    return data


if __name__=="__main__":

    data=gather_basic_treebank_info()
    data=get_udpipe_baseline_numbers(data)

    with open("udv2.2_treebank_info.json", "wt") as f:
        json.dump(data, f, indent=2)
