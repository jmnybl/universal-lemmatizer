import sys
from collections import Counter

ID,FORM,LEMMA,UPOS,XPOS,FEATS,HEAD,DEPREL,DEPS,MISC=range(10)

def collect_lemmas(training_file):

    d={} # key: (form, upos, feats), value: counter of lemmas
    for line in open(training_file, "rt"):
        line=line.strip()
        if not line or line.startswith("#"):
            continue
        cols=line.split("\t")
        key=(cols[FORM],cols[UPOS],cols[FEATS])
        if key not in d:
            d[key]=Counter()
        d[key].update([cols[LEMMA]])

    # now filter the dictionary so that each (form, upos, feats) only keeps the most common lemma
    filtered_d={}
    for key,counter in d.items():
        lemma,count=counter.most_common(1)[0]
        filtered_d[key]=lemma
        #if len(counter.most_common())>1:
            #print("Ambigious lemma:", key,counter,file=sys.stderr)

    return filtered_d

def lemmatize(test_file, lemma_dict):

    lines=[]
    for line in open(test_file, "rt"):
        line=line.strip()
        if not line or line.startswith("#"):
            lines.append(line)
            continue
        cols=line.split("\t")
        key=(cols[FORM],cols[UPOS],cols[FEATS])
        if key not in lemma_dict:
            cols[LEMMA]=cols[FORM]
        else:
            cols[LEMMA]=lemma_dict[key]
        lines.append("\t".join(cols))

    return lines


def write_file(data, f):
    for line in data:
        print(line,file=f)

def main(args):

    if args.training_file:
        lemmatizer_dictionary=collect_lemmas(args.training_file)
    elif args.load_dict:
        import pickle
        with open(args.load_dict, "rb") as f:
            lemmatizer_dictionary=pickle.load(f)
    else:
        print("Give either --training_file or --load_dict.", file=sys.stderr)
        sys.exit(1)
    
    if args.save_dict:
        import pickle
        with open(args.save_dict, "wb") as f:
            pickle.dump(lemmatizer_dictionary, f)

    if args.test_file:
        data=lemmatize(args.test_file, lemmatizer_dictionary)

    write_file(data,sys.stdout)


if __name__=="__main__":
    import argparse
    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('--training_file', type=str, help='Training data file (conllu)')
    argparser.add_argument('--test_file', type=str, help='Test data file (conllu)')
    argparser.add_argument('--save_dict', type=str, help='File name to save the dictionary (json)')
    argparser.add_argument('--load_dict', type=str, help='File name to save the dictionary (json)')
    args = argparser.parse_args()

    main(args)
