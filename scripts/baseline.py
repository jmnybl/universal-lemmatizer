import sys
import json
import glob
import os.path
from collections import Counter


ID,FORM,LEMMA,UPOS,XPOS,FEATS,HEAD,DEPREL,DEPS,MISC=range(10)

def collect_dictionaries(train_file):

    tag_based_d={} # key: (form, upos, feats), value: counter of lemmas # TODO: XPOS??
    form_based_d={} # key: form, value: counter of lemmas
    empty_lemmas=0
    equals_form=0
    tokens=0

    for line in open(train_file, "rt"):
        line=line.strip()
        if not line or line.startswith("#"):
            continue
        cols=line.split("\t")
        if "-" in cols[ID] or "." in cols[ID]: # skip multiword and null nodes
            continue
        tokens+=1
        if cols[LEMMA]=="_" and cols[FORM]!="_": # something wrong with lemma, skipping
            empty_lemmas+=1
            continue
        if cols[FORM]==cols[LEMMA]:
            equals_form+=1
        key=(cols[FORM],cols[UPOS],cols[XPOS],cols[FEATS])
        if key not in tag_based_d:
            tag_based_d[key]=Counter()
        tag_based_d[key].update([cols[LEMMA]])
        if cols[FORM] not in form_based_d:
            form_based_d[cols[FORM]]=Counter()
        form_based_d[cols[FORM]].update([cols[LEMMA]])

    return tag_based_d, form_based_d, round(empty_lemmas/tokens*100,2), round(equals_form/tokens*100,2)

def count_ambiguous(d):
    #count how many unique/running words are ambigious in given dictionary

    if len(d)==0: # must mean that treebank does not include lemmas
        return 0.0, 0.0

    running_tokens_total=0
    running_tokens_amb=0
    uniq_tokens_total=0
    uniq_tokens_amb=0
    for key,counter in d.items():
        uniq_tokens_total+=1
        running_tokens_total+=len(list(counter.elements()))
        if len(counter.items())>1:
            # ambigious key
            uniq_tokens_amb+=1
            running_tokens_amb+=len(list(counter.elements()))

    return round((uniq_tokens_amb/uniq_tokens_total)*100,2), round((running_tokens_amb/running_tokens_total)*100,2)

def filter(d):
    # keep only most common for each ambigious key

    # now filter the dictionary so that each (form, upos, feats) only keeps the most common lemma
    filtered={}
    for key,counter in d.items():
        lemma,count=counter.most_common(1)[0]
        filtered[key]=lemma

    return filtered

def baseline_accuracy(tag_dictionary,form_dictionary,test_data):

    

    total_words=0
    correct_tag=0
    correct_form=0

    for line in open(test_data, "rt"):
        line=line.strip()
        if not line or line.startswith("#"):
            continue
        total_words+=1
        cols=line.split("\t")
        form=cols[FORM]
        key=(cols[FORM],cols[UPOS],cols[XPOS],cols[FEATS])
        lemma=cols[LEMMA]
        if key in tag_dictionary and tag_dictionary[key]==lemma:
            correct_tag+=1
        if key not in tag_dictionary and form==lemma:
            correct_tag+=1
        if form in form_dictionary and form_dictionary[form]==lemma:
            correct_form+=1
        if form not in form_dictionary and form==lemma:
            correct_form+=1

    return round(correct_tag/total_words*100,2), round(correct_form/total_words*100,2)


def main(treebank_code, data_path, results):

    if treebank_code not in results:
        results[treebank_code]={}

    training_data=glob.glob(os.path.join(data_path,"*",treebank_code+"-ud-train.conllu"))[0]
    if not os.path.isfile(training_data):
        results[treebank_code]["training data"]="no"
        return results

    test_data=glob.glob(os.path.join(data_path,"*",treebank_code+"-ud-test.conllu"))[0]

    tag_dict, form_dict, empty, equal = collect_dictionaries(training_data)

    tag_uniq, tag_running = count_ambiguous(tag_dict)

    form_uniq, form_running = count_ambiguous(form_dict)

    filtered_tag_dict = filter(tag_dict)
    filtered_form_dict = filter(form_dict)


    tag_acc, form_acc = baseline_accuracy(filtered_tag_dict, filtered_form_dict, test_data)

    

    results[treebank_code]["dictionary baseline (form+tag) accuracy on test"]=tag_acc
    results[treebank_code]["dictionary baseline (form) accuracy on test"]=form_acc
    results[treebank_code]["uniq tokens (form+tag) ambiguous on train"]=tag_uniq
    results[treebank_code]["running tokens (form+tag) ambiguous on train"]=tag_running
    results[treebank_code]["uniq tokens (form) ambiguous on train"]=form_uniq
    results[treebank_code]["running tokens (form) ambiguous on train"]=form_running
    results[treebank_code]["empty lemmas"]=empty
    results[treebank_code]["form equals lemma"]=equal

    return results

if __name__=="__main__":
    import argparse
    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('--treebank', type=str, default="", help='Treebank code (default="", all)')
    argparser.add_argument('--data_path', type=str, help='Data directory for treebanks')
    args = argparser.parse_args()

    if args.treebank!="":
        treebanks=[args.treebank]
    else:
       treebanks="af_afribooms ar_padt bg_btb bxr_bdt ca_ancora cs_cac cs_fictree cs_pdt cu_proiel da_ddt de_gsd el_gdt en_ewt en_gum en_lines es_ancora et_edt eu_bdt fa_seraji fi_ftb fi_tdt fr_gsd fr_sequoia fr_spoken fro_srcmf ga_idt gl_ctg gl_treegal got_proiel grc_perseus grc_proiel he_htb hi_hdtb hr_set hsb_ufal hu_szeged hy_armtdp id_gsd it_isdt it_postwita ja_gsd kk_ktb kmr_mg ko_gsd ko_kaist la_ittb la_perseus la_proiel lv_lvtb nl_alpino nl_lassysmall no_bokmaal no_nynorsk no_nynorsklia pl_lfg pl_sz pt_bosque ro_rrt ru_syntagrus ru_taiga sme_giella sr_set sk_snk sl_ssj sl_sst sv_lines sv_talbanken tr_imst ug_udt uk_iu ur_udtb vi_vtb zh_gsd".split(" ")

    stats={}
    for tb in treebanks:
        print(tb)
        stats=main(tb, args.data_path, stats)


    # make copy of old json
    if os.path.isfile("udv2.2_treebank_info.json"):
        os.system("cp udv2.2_treebank_info.json udv2.2_treebank_info.json.copy")
        with open("udv2.2_treebank_info.json","rt") as f:
            data=json.load(f)

    for key in data.keys():
        treebank_code=data[key]["treebank_code"]
        if treebank_code in stats:
            for new_key in stats[treebank_code]:
                data[key][new_key]=stats[treebank_code][new_key]

    # merge
    with open("udv2.2_treebank_info.json","wt") as f:
        json.dump(data, f, indent=2)




    
