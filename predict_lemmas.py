#!/usr/bin/env python
from __future__ import division, unicode_literals
import argparse
import io
import sys
import os
import select

from prepare_data import read_conllu, transform_token, detransform_token, ID

sys.path.insert(0,os.getcwd()+"/OpenNMT-py") # could be replaced with symlinks

from onmt.translate.Translator import make_translator

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import onmt.opts
import sys


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




def main(opt):

    f_input=io.StringIO() # make virtual files to collect the transformed input and output
    f_output=io.StringIO()


    translator = make_translator(opt, report_score=True, out_file=f_output) # always output to virtual file


    if opt.src!="":
        corpus_file = open(opt.src, "rt", encoding="utf-8")
    else: 
        corpus_file = sys.stdin

    if opt.output!="":
        real_output_file=open(opt.output, "wt", encoding="utf-8")
    else:
        real_output_file=sys.stdout

    for batch in nonblocking_batches(f=corpus_file):
        original_sentences=[]
        for (comm, sent) in read_conllu(batch.split("\n")):
            original_sentences.append((comm, sent))
            for token in sent:
                if "-" in token[ID]: # multiword token line, not supposed to be analysed
                    continue
                form, _ = transform_token(token)
                print(form, file=f_input, flush=True)

        # run lemmatizer
        f_input.seek(0) # beginning of the virtual file
        translator.translate(opt.src_dir, f_input, opt.tgt,
                         opt.batch_size, opt.attn_debug)

        # collect lemmas from virtual output file, transform and inject to conllu
        f_output.seek(0)
        for comm, sent in original_sentences:
            for c in comm:
                print(c, file=real_output_file)
            for cols in sent:
                if "-" in cols[ID]: # multiword token line, not supposed to be analysed
                    print("\t".join(t for t in cols), file=real_output_file, flush=True)
                predicted_lemma=f_output.readline().strip()
                cols, token = detransform_token(cols, predicted_lemma)
                print("\t".join(t for t in cols), file=real_output_file, flush=True)
            print(file=real_output_file, flush=True)

        f_input=io.StringIO() # clear virtual files
        f_output=io.StringIO()
        translator.out_file=f_output

    if opt.src!="":
        corpus_file.close()
    if opt.output!="":
        real_output_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    main(opt)
