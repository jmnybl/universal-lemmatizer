#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import configargparse

import io
import sys
import os
import select


from prepare_data import read_conllu, transform_token, detransform_token, detransform_string, ID, FORM, LEMMA, UPOS, XPOS, FEAT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "OpenNMT-py"))


from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts



def nonblocking_batches(f=sys.stdin,timeout=0.2,batch_lines=5000):
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
        parser = configargparse.ArgumentParser(
        description='translate.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,conflict_handler="resolve")
        onmt.opts.config_opts(parser)
        onmt.opts.add_md_help_argument(parser)
        onmt.opts.translate_opts(parser)

        # rewrite src/output arguments because we do not want these to be required anymore (default is empty, use stdin/stdout)
        parser.add_argument("--src", "-src", default="", help="""Source sequence to decode (one line per
                       sequence)""")
        parser.add_argument("--output", "-output", default="", help="""Path to output the predictions (each line will
                       be the decoded sequence""")

        if not args: # take arguments from sys.argv (this must be called from the main)
            self.opt = parser.parse_args()
        else:
            self.opt = parser.parse_args(args)


        # make virtual files to collect the predicted output (not actually needed but opennmt still requires this)
        self.f_output=io.StringIO()

        self.translator = build_translator(self.opt, report_score=True, out_file=self.f_output)

        self.localcache={} #tokendata -> lemma  #remembered by this process, lost thereafter


    def lemmatize_batch(self, data_batch):
        """ Lemmatize one data batch """

        submitted=set() #set of submitted tokens
        submitted_tdata=[] #list of token data entries submitted for lemmatization

        # lemmatize data_batch
        original_sentences=[]
        translate_input=[]
        token_counter=0
        for (comm, sent) in read_conllu(data_batch.split("\n")):
            original_sentences.append((comm, sent))
            for token in sent:
                if "-" in token[ID]: # multiword token line, not supposed to be analysed
                    continue
                token_counter+=1
                if token[LEMMA]!="_": # already filled in for example by another module, do not process
                    continue
                token_data=(token[FORM],token[UPOS],token[XPOS],token[FEAT])
                if token_data not in self.localcache and token_data not in submitted:
                    submitted.add(token_data)
                    submitted_tdata.append(token_data)
                    form, _ = transform_token(token)
                    translate_input.append(form)
        print(" >>> {}/{} unique tokens submitted to lemmatizer".format(len(submitted_tdata),token_counter),file=sys.stderr)
        # run lemmatizer if everything is not in cache
        if len(submitted_tdata)>0:

            scores, predictions=self.translator.translate(src_data_iter=translate_input, batch_size=self.opt.batch_size)
            self.f_output.truncate(0) # clear this to prevent eating memory

            lemm_output=[l[0] for l in predictions]
            for tdata,predicted_lemma in zip(submitted_tdata,lemm_output):
                predicted_lemma=detransform_string(predicted_lemma.strip())
                self.localcache[tdata]=predicted_lemma
        output_lines=[]
        for comm, sent in original_sentences:
            for c in comm:
                output_lines.append(c)
            for cols in sent:
                if "-" in cols[ID] or cols[LEMMA]!="_": # multiword token line or lemma already predicted, not supposed to be analysed
                    output_lines.append("\t".join(t for t in cols))
                    continue
                token_data=(cols[FORM],cols[UPOS],cols[XPOS],cols[FEAT])
                if token_data in self.localcache:
                    plemma=self.localcache[token_data]
                else:
                    assert False, ("Missing lemma", token_data)
                if plemma.strip()=="":
                    plemma="_" # make sure not to output empty lemma
                cols[LEMMA]=plemma
                output_lines.append("\t".join(t for t in cols))
            output_lines.append("")


        return "\n".join(output_lines)+"\n"

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
        print(lemmatized_batch, file=real_output_file, flush=True, end="")


    # close files if needed
    if lemmatizer.opt.src!="":
        corpus_file.close()
    if lemmatizer.opt.output!="":
        real_output_file.close()



if __name__ == "__main__":

    main()
