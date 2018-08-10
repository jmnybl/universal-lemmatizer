import sys
import os
import json
from random import shuffle
from prepare_data import read_conllu

ID,FORM,LEMMA,UPOS,XPOS,FEAT,HEAD,DEPREL,DEPS,MISC=range(10)

def read_word_frequencies(filename):
    with open(filename, "rt") as f:
        data=json.load(f)
    return [(key,val) for key,val in sorted(data.items(), key=lambda x:x[1], reverse=True)]

def read_treebank_words(training_file):

    words=set()
    f=open(training_file, "rt")
    for comm, sent in read_conllu(f):
        for token in sent:
            words.add(token[FORM])
    return words

def read_transducer(transducer_file):

    if transducer_file.endswith(".gz"):
        f=gzip.open(transducer_file, "rt")
    else:
        f=open(transducer_file, "rt")

    all_readings={}

    word=None
    block=[]
    readings={}

    for line in f:
        line=line.strip()
        if not line:
            if readings:
                all_readings[word]=readings
                #print("saving:",form, readings)
                word=None
                readings={}
                block=[]
                continue
        else:
            try:
                form,lemma,upos,feat=line.split("\t")
            except:
                print("Something weird:",line, file=sys.stderr)
                continue
            if upos=="_":
                #print("skipping bad reading:",upos,feat,lemma)
                continue
            if feat=="_" and (upos=="NOUN" or upos=="VERB"): # bad reading
                continue
            if (upos,feat) in readings and readings[(upos,feat)]!=lemma: #ambiguous lemma, remove this key from the dictionary and block the reading
                readings.pop((upos,feat), None)
                block=[(upos,feat)]
                continue
            if (upos,feat) in block:
                continue
            readings[(upos,feat)]=lemma
            word=form

    return all_readings


def collect_readings(transducer, word_freq, treebank_data, max_words):

    all_readings=read_transducer(transducer)
    word_frequencies=read_word_frequencies(word_freq)
    print(word_frequencies[:100],file=sys.stderr)
    treebank_words=read_treebank_words(treebank_data)

    # now sample 4K most frequent words from the transducer readings which does not already appear in the treebank training data
    data=[]
    counter=0
    for word, count in word_frequencies:
        if word in treebank_words:
            continue
        if word not in all_readings:
            continue
        readings=all_readings[word]
        keys=[k for k,val in readings.items()]
        examples=[]
        for key in keys: # use all possible readings (words with ambigious lemma readings are already dropped)
            upos,feat=key
            example=(word,upos,feat,readings[(upos,feat)])
            data.append(example)
        counter+=1
        if counter>=max_words:
            break

    return data


def create_data(transducer, word_freq, treebank_data, max_words, extra_tag):

    initial_data=collect_readings(transducer, word_freq, treebank_data, max_words)
    data=[]
    for form, upos, feats, lemma in initial_data:

        lemma=" ".join(c if c!=" " else "$@@$" for c in lemma) # replace whitespace with $@@$
        wordform=" ".join(c if c!=" " else "$@@$" for c in form)
        
        # note: cannot add XPOS
        tags=[]
        if extra_tag!="":
            tags.append(extra_tag)
        tags.append("UPOS="+upos)
        for feat in feats.split("|"):
            if feat=="_":
                tags.append("FEAT="+feat)
            else:
                tags.append(feat)
        tags=" ".join(tags)
        wordform=" ".join([wordform, tags])
        data.append((wordform, lemma))

    return data

if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    
    g.add_argument('--transducer', type=str, help='Name of the transducer output file')
    g.add_argument('--output', type=str, help='Output file name, will create file extentions .input and .output')
    g.add_argument('--word_freq', type=str, help='Word frequency list')
    g.add_argument('--training_data', type=str, default="", help='Treebank training data file')
    g.add_argument('--max_words', type=int, default=4000, help='Treebank training data file')
    g.add_argument('--extra_tag', type=str, default="", help='extra tag, for example mark autoencoding training examples')
    
    args = parser.parse_args()

    data=create_data(args.transducer, args.word_freq, args.training_data, args.max_words, args.extra_tag, args.min_freq, args.weighted)
    for form , lemma in data:
        print(form, lemma, sep="\t")



