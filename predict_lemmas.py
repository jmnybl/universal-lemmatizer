#!/usr/bin/env python
from __future__ import division, unicode_literals
import argparse
import io
import sys
import os
import select

from prepare_data import read_conllu, transform_token, detransform_token, detransform_string, ID, FORM, LEMMA, UPOS, XPOS, FEAT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "OpenNMT-py"))

from onmt.translate.Translator import make_translator

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import onmt.opts


def nonblocking_batches(f=sys.stdin,timeout=0.2,batch_lines=1000):
    """Yields batches of the input conllu (as string), always ending with an empty line.
    Batch is formed when at least batch_lines are read, or when no input is seen in timeour seconds
    Stops yielding when f is closed"""
    line_buffer=[]
    while True:
        ready_to_read=select.select([f], [], [], timeout)[0] #check whether f is ready to be read, wait at least timeout (otherwise we run a crazy fast loop)
        if not ready_to_read:
            # Stdin is not ready, yield what we've got, if anything
            if line_buffer:
                yield "".join(line_buffer)
                line_buffer=[]
            continue #next try
        
        # f is ready to read!
        # Since we are reading conll, we should always get stuff until the next empty line, even if it means blocking read
        while True:
            line=f.readline()
            if not line: #End of file detected --- I guess :D
                if line_buffer:
                    yield "".join(line_buffer)
                    return
            line_buffer.append(line)
            if not line.strip(): #empty line
                break

        # Now we got the next sentence --- do we have enough to yield?
        if len(line_buffer)>batch_lines:
            yield "".join(line_buffer) #got plenty
            line_buffer=[]


class Lemmatizer(object):

    def __init__(self, args=None):
        # init lemmatizer model
        parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        onmt.opts.add_md_help_argument(parser)
        onmt.opts.translate_opts(parser)

        if not args: # take arguments from sys.argv (this must be called from the main)
            self.opt = parser.parse_args()
        else:
            self.opt = parser.parse_args(args)

        # make virtual files to collect the transformed input and output
        self.f_input=io.StringIO() 
        self.f_output=io.StringIO()

        self.translator = make_translator(self.opt, report_score=True, out_file=self.f_output) # always output to virtual file
        self.cache={} #tokendata -> lemma  #comes pre-computed with the model
        try:
            if self.opt.lemma_cache:
                with gzip.open(self.opt.lemma_cache,"rt") as f:
                    self.cache=json.load(f)
        except AttributeError:
            pass
        self.localcache={} #tokendata -> lemma  #remembered by this process, lost thereafter
        
    def lemmatize_batch(self, data_batch):

        submitted=set() #set of submitted tokens
        submitted_tdata=[] #list of token data entries submitted for lemmatization

        # lemmatize data_batch
        original_sentences=[]
        token_counter=0
        for (comm, sent) in read_conllu(data_batch.split("\n")):
            original_sentences.append((comm, sent))
            for token in sent:
                if "-" in token[ID]: # multiword token line, not supposed to be analysed
                    continue
                token_counter+=1
                token_data=(token[FORM],token[UPOS],token[XPOS],token[FEAT])
                if token_data not in self.cache and token_data not in submitted and token_data not in self.localcache:
                    submitted.add(token_data)
                    submitted_tdata.append(token_data)
                    form, _ = transform_token(token)
                    print(form, file=self.f_input)
        self.f_input.flush()
        print(" >>> {}/{} submitted to lemmatizer, rest in cache".format(len(submitted_tdata),token_counter),file=sys.stderr)
        # run lemmatizer
        self.f_input.seek(0) # beginning of the virtual file
        self.translator.translate(self.opt.src_dir, self.f_input, self.opt.tgt,
                         self.opt.batch_size, self.opt.attn_debug) # TODO how to deal with missing opt

        # collect lemmas from virtual output file, transform and inject to conllu
        self.f_output.seek(0)
        lemmatized_batch={} #token-data -> lemma
        lemm_output=list(self.f_output.readlines())
        for tdata,predicted_lemma in zip(submitted_tdata,lemm_output):
            predicted_lemma=detransform_string(predicted_lemma.strip())
            self.localcache[tdata]=predicted_lemma
        output_lines=[]
        for comm, sent in original_sentences:
            for c in comm:
                output_lines.append(c)
            for cols in sent:
                if "-" in cols[ID]: # multiword token line, not supposed to be analysed
                    output_lines.append("\t".join(t for t in cols))
                    continue
                token_data=(cols[FORM],cols[UPOS],cols[XPOS],cols[FEAT])
                if token_data in self.cache:
                    plemma=self.cache[token_data]
                elif token_data in self.localcache:
                    plemma=self.localcache[token_data]
                else:
                    assert False, ("Missing lemma", token_data)
                cols[LEMMA]=plemma
                output_lines.append("\t".join(t for t in cols))
            output_lines.append("")

        self.f_input=io.StringIO() # clear virtual files
        self.f_output=io.StringIO()
        self.translator.out_file=self.f_output

        return "\n".join(output_lines)

def main():

    # init and load models
    lemmatizer=Lemmatizer()

    # input file
    if lemmatizer.opt.src!="":
        corpus_file = open(lemmatizer.opt.src, "rt", encoding="utf-8")
    else: 
        corpus_file = sys.stdin

    # output file
    if lemmatizer.opt.output!="":
        real_output_file=open(lemmatizer.opt.output, "wt", encoding="utf-8")
    else:
        real_output_file=sys.stdout

    # lemmatize
    for batch in nonblocking_batches(f=corpus_file):

        lemmatized_batch=lemmatizer.lemmatize_batch(batch)
        print(lemmatized_batch, file=real_output_file, flush=True)


    # close files if needed
    if lemmatizer.opt.src!="":
        corpus_file.close()
    if lemmatizer.opt.output!="":
        real_output_file.close()



if __name__ == "__main__":

    main()
