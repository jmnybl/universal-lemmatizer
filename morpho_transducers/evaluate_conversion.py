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

def transducer_reader(transducer_file):
    """ Yield one wordform with all readings at once from the transducer file. """
    readings=[]
    if transducer_file.endswith(".gz"):
        import gzip
        f=gzip.open(transducer_file,"rt",encoding="utf-8")
    else:
        f=open(transducer_file,"rt",encoding="utf-8")
    for line in f:
        line=line.strip()
        if not line:
            if readings:
                yield readings
                readings=[]
            continue
        if line.split("\t") not in readings:
            readings.append(line.split("\t"))

def read_transducer(transducer_file, max_words=0):
    """ Return the transducer output as dictionary where key: wordform, value: list of readings """
    transducer_words={} # key: word, value: list of (lemma, upos, tags) tuples
    for readings in transducer_reader(transducer_file):
        word=readings[0][0]
        if word in transducer_words: # transducer output is not unique words, so skip if it's already stored
            continue
        all_readings=[]
        for item in readings:
            if item[1].startswith("*") and item[1].endswith("$"): # unrecognized # TODO: should this be here or in the convertion script?
                continue
            all_readings.append(tuple(item[1:]))# remove word from the list as it is the key
        if all_readings:
            transducer_words[word]=all_readings
        if max_words!=0 and len(transducer_words)>max_words:
            break
    return transducer_words



def lemma_recall(gold_file, transducer_readings, lowercase=False):
    """ How many treebank lemmas are found from the transducer output. """
    total_words=0 # how many words is in the treebank
    unrecognized_words=0 # how many are not recognized by the transducer
    matching_lemmas=0 # how many times treebank lemma is found among the transducer readings
    with open(gold_file, "rt", encoding="utf-8") as f:
        for comm, sent in conllu_reader(f):
            for token in sent:
                word=token[FORM]
                total_words+=1
                if word not in transducer_readings:
                    unrecognized_words+=1
                    continue
                if lowercase:
                    treebank_lemma=token[LEMMA].lower() # gold lemma
                    transducer_lemmas=set([l.lower() for l,p,t in transducer_readings[word]])
                else:
                    treebank_lemma=token[LEMMA] # gold lemma
                    transducer_lemmas=set([l for l,p,t in transducer_readings[word]])
                if treebank_lemma in transducer_lemmas:
                    matching_lemmas+=1
    return total_words, unrecognized_words, matching_lemmas, matching_lemmas/total_words if (matching_lemmas!=0 and total_words!=0) else 0 # return total_words, unrecognized, mathing, recall


def validate_features(features):
    """ Validate that features are in the correct order, if not conversion does something wrong. These lines are copied from the UD validation tool."""
    feat_list=features.split(u"|")
    #the lower() thing is to be on the safe side, since all features must start with [A-Z0-9] anyway
    if [f.lower() for f in feat_list]!=sorted(f.lower() for f in feat_list):
        print(u"Morphological features must be sorted: '{x}'".format(x=features), file=sys.stderr)


def oracle_full_match(gold_file, transducer_readings, lowercase=False):
    """ Count oracle full match, i.e. how many times treebank (lemma,upos,features) triplet can be found from the transducer output exactly as it is in the treebank."""
    total_words=0 # how many words is in the treebank
    unrecognized_words=0 # how many are not recognized by the transducer
    full_match=0 # how many times treebank (lemma,upos,features) triplet is found among the transducer readings
    with open(gold_file, "rt", encoding="utf-8") as f:
        for comm, sent in conllu_reader(f):
            for token in sent:
                word=token[FORM]
                total_words+=1
                if word not in transducer_readings:
                    unrecognized_words+=1
                    continue
                if lowercase:
                    treebank_reading=(token[LEMMA].lower(),token[UPOS],token[FEAT]) # gold triplet
                    transducer_reading=set([(l.lower(),p,t) for l,p,t in transducer_readings[word]])
                else:
                    treebank_reading=(token[LEMMA],token[UPOS],token[FEAT]) # gold triplet
                    transducer_reading=set([(l,p,t) for l,p,t in transducer_readings[word]])
                for r in transducer_reading:
                    validate_features(r[-1])
                if treebank_reading in transducer_reading:
                    full_match+=1

    return total_words, unrecognized_words, full_match, full_match/total_words if (full_match!=0 and total_words!=0) else 0 # return recall

    

def oracle_full_match_without_lemma(gold_file, transducer_readings, lowercase=False):
    """ Count how many times a word is found from the transducer output with correct tags, not caring about the lemma being correct or incorrect. We can directly compare this to full_match and from the difference say whether lemmas are cause problem or not."""
    total_words=0 # how many words is in the treebank
    unrecognized_words=0 # how many are not recognized by the transducer
    full_match=0 # how many times treebank (lemma,upos,features) triplet is found among the transducer readings
    with open(gold_file, "rt", encoding="utf-8") as f:
        for comm, sent in conllu_reader(f):
            for token in sent:
                word=token[FORM]
                total_words+=1
                if word not in transducer_readings:
                    unrecognized_words+=1
                    continue
                if lowercase:
                    print("Warning! Lowercasing does not have effect in full match without lemma, lowercasibng parameter is ignored!")
                treebank_reading=(token[UPOS],token[FEAT]) # gold tuplet (no lemma)
                transducer_reading=set([(p,t) for l,p,t in transducer_readings[word]])
                for r in transducer_reading:
                    validate_features(r[-1])
                if treebank_reading in transducer_reading:
                    full_match+=1

    return total_words, unrecognized_words, full_match, full_match/total_words if (full_match!=0 and total_words!=0) else 0


def tag_recall(gold_file, transducer_readings, lowercase=False):
    """ If correct lemma is found, count how many times the tags are correct (exact match), or the transducer tags are a subset of the gold tags (can be less but not extra). Tries to prove that the conversion strives for high precision and low recall. If word or correct lemma is not found, then just pass (do not punish)."""
    total_words=0 # how many words is in the treebank
    unrecognized_words=0 # how many are not recognized by the transducer
    lemma_found=0 # how many times treebank lemma is found among the transducer readings
    tags_exact=0 # how many times the tags are exact match if the treebank lemma is found from the transducer
    tags_subset=0 # how many times transducer tags are a subset of the treebank tags if the treebank lemma is found from the transducer
    with open(gold_file, "rt", encoding="utf-8") as f:
        for comm, sent in conllu_reader(f):
            for token in sent:
                word=token[FORM]
                treebank_lemma=token[LEMMA]
                total_words+=1
                if word not in transducer_readings:
                    unrecognized_words+=1
                    continue
                if lowercase:
                    print("Warning! tag recall with lowercasing not implemented.")
                    sys.exit()
                treebank_reading=set([token[UPOS]]+[t for t in token[FEAT].split("|") if t!="_"]) # gold tags
                transducer_reading=[set([p]+[t for t in t.split("|") if t!="_"]) for l,p,t in transducer_readings[word] if l==treebank_lemma] # list of tag sets where filtered with lemma
                if len(transducer_reading)==0:
                    continue # correct lemma not found
                lemma_found+=1
                if treebank_reading in transducer_reading:
                    tags_exact+=1
                    tags_subset+=1
                    continue
                # not exact match, check whether subset
                for tagset in transducer_reading:
                    if tagset.issubset(treebank_reading):
                        tags_subset+=1
                        break

    return total_words, unrecognized_words, lemma_found, tags_exact, tags_subset, tags_exact/lemma_found if (tags_exact!=0 and lemma_found!=0) else 0, tags_subset/lemma_found if (tags_subset!=0 and lemma_found!=0) else 0 # return recall

def evaluate(args):

    treebank_words=read_treebank(args) # key: word, value: list of (lemma, upos, tags) tuples

    transducer_words=read_transducer(args.transducer, args.max_words)
    total, unrecognized, mathing, recall=lemma_recall(args.treebank, transducer_words)
    print("Total words:", total)
    print("Words not recognized by the transducer:", unrecognized, "({x}%)".format(x=unrecognized/total*100))
    print()
    print("Lemma recall without lowercasing:",recall)
    
    total, unrecognized, mathing, recall=lemma_recall(args.treebank, transducer_words, lowercase=True)
    print("Lemma recall with lowercasing:",recall)
    print()

    total, unrecognized, full_match, recall=oracle_full_match(args.treebank, transducer_words)
    print("Oracle full match without lowercasing:",recall)
    total, unrecognized, full_match, recall=oracle_full_match(args.treebank, transducer_words, lowercase=True)
    print("Oracle full match with lowercasing:",recall)
    print()

    total, unrecognized, full_match, recall=oracle_full_match_without_lemma(args.treebank, transducer_words)
    print("Oracle full match without lemma:",recall)
    print()

    total, unrecognized, lemma_found, exact, subset, exact_recall, subset_recall=tag_recall(args.treebank, transducer_words)
    print("Tags exact match (when lemma correct):", exact_recall)
    print("Tags subset match (when lemma correct):", subset_recall)
    print()


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    
    g.add_argument('--treebank', type=str, help='Input treebank file name')
    g.add_argument('--transducer', type=str, help='Input transducer output file name')
    g.add_argument('--max_words', type=int, default=0, help='How many words to read from converted transducer output, default: 0 (all)')
    
    args = parser.parse_args()

    evaluate(args)
