import sys
from collections import Counter

ID,FORM,LEMMA,UPOS,XPOS,FEATS,HEAD,DEPREL,DEPS,MISC=range(10)

def collect_lemmas(training_file, min_count):

    d={} # key: (form, upos, feats), value: counter of lemmas
    for line in open(training_file, "rt"):
        line=line.strip()
        if not line or line.startswith("#"):
            continue
        cols=line.split("\t")
        key=(cols[FORM],cols[UPOS],cols[XPOS],cols[FEATS])
        if key not in d:
            d[key]=Counter()
        d[key].update([cols[LEMMA]])

    # now filter the dictionary so that each (form, upos, feats) only keeps the most common lemma
    filtered_d={}
    for key,counter in d.items():
        lemma,count=counter.most_common(1)[0]
        if count<min_count:
            continue
        filtered_d[key]=lemma
        #if len(counter.most_common())>1:
            #print("Ambigious lemma:", key,counter,file=sys.stderr)

    return filtered_d

def lemmatize(test_file, lemma_dict):

    out_of_vocab=0
    correct=0
    total=0
    lines=[]
    for line in open(test_file, "rt"):
        line=line.strip()
        if not line or line.startswith("#"):
            lines.append(line)
            continue
        cols=line.split("\t")
        key=(cols[FORM],cols[UPOS],cols[XPOS],cols[FEATS])
        total+=1
        if key not in lemma_dict:
            cols[LEMMA]=cols[FORM]
            out_of_vocab+=1
        else:
            if cols[LEMMA]==lemma_dict[key]:
                correct+=1
            cols[LEMMA]=lemma_dict[key]
        lines.append("\t".join(cols))

    print("Lemmatized {x} words.".format(x=total),file=sys.stderr)
    print("Out-of-vocab words: {x} ({y}%)".format(x=out_of_vocab, y=out_of_vocab/total*100),file=sys.stderr)
    print("Accuracy: {x}%".format(x=correct/total*100),file=sys.stderr)
    print("Accuracy of known words: {x}%".format(x=correct/(total-out_of_vocab)*100),file=sys.stderr)
    

    return lines

def save_dict(args, data):
    """ save dictionary as tsv file """
    with open(args.save_dict, "wt") as f:
        for (form, upos, xpos, feats), lemma in data.items():
            print("\t".join([form, upos, xpos, feats,lemma]),file=f)

def load_dict(args):
    """ save dictionary as tsv file """
    d={}
    with open(args.load_dict, "rt") as f:
        for line in f:
            form, upos, xpos, feats, lemma = line.strip.split("\t")
            d[(form, upos, xpos, feats)]=lemma
    return d


def write_file(data, f):
    for line in data:
        print(line,file=f)

def main(args):

    if args.training_file:
        lemmatizer_dictionary=collect_lemmas(args.training_file, args.min_freq)
    elif args.load_dict:
        lemmatizer_dictionary=load_dict(args)
    else:
        print("Give either --training_file or --load_dict.", file=sys.stderr)
        sys.exit(1)
    
    if args.save_dict:
        save_dict(args, lemmatizer_dictionary)

    if args.test_file:
        data=lemmatize(args.test_file, lemmatizer_dictionary)
        write_file(data,sys.stdout)


if __name__=="__main__":
    import argparse
    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('--training_file', type=str, help='Training data file (conllu)')
    argparser.add_argument('--test_file', type=str, help='Test data file (conllu)')
    argparser.add_argument('--save_dict', type=str, help='File name to save the dictionary (tsv)')
    argparser.add_argument('--load_dict', type=str, help='File name to load the dictionary (tsv)')
    argparser.add_argument('--min_freq', type=int, default=0, help='Minimum frequency for token to be preserved in the dictionary (default:0, save all)')
    args = argparser.parse_args()

    main(args)
