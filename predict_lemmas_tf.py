"""Main library entrypoint."""


import io
import os
import sys
import random
import argparse
import select
print("AAAAAAAA", file=sys.stderr)

import numpy as np
import tensorflow as tf

from tensorflow.python.estimator.util import fn_args

from google.protobuf import text_format

# trying to monkey path opennmt text_inputter
import opennmt
class TextInputter(opennmt.inputters.text_inputter.TextInputter):

    def make_dataset(self, data_file):
       print("Running custom function", file=sys.stderr)
       sys.exit()
       if isinstance(data_file, str):
          return tf.data.TextLineDataset(data_file)
       else:
           return tf.data.Dataset.from_tensor_slices(data_file)

opennmt.inputters.text_inputter.TextInputter = TextInputter
print("Function overwritten", file=sys.stderr)
from opennmt.inputters.text_inputter import TextInputter
print("Function loaded", file=sys.stderr)

from opennmt.utils import hooks, checkpoint
from opennmt.utils.evaluator import external_evaluation_fn
from opennmt.utils.misc import extract_batches, print_bytes

from opennmt.config import load_model, load_config


# TODO this will be fixed when incorporated with parser pipeline
sys.path.insert(0,"/home/jmnybl/git_checkout/universal-lemmatizer")
from prepare_data import read_conllu, transform_token, detransform_token, detransform_string, ID, FORM, LEMMA, UPOS, XPOS, FEAT




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
  """Class for managing training, inference, and export. It is mostly a
  wrapper around ``tf.estimator.Estimator``.
  """

  def __init__(self,
               model,
               config,
               seed=None,
               num_devices=1,
               gpu_allow_growth=False,
               session_config=None):
    """Initializes the runner parameters.

    Args:
      model: A :class:`opennmt.models.model.Model` instance to run.
      config: The run configuration.
      seed: The random seed to set.
      num_devices: The number of devices (GPUs) to use for training.
      gpu_allow_growth: Allow GPU memory to grow dynamically.
      session_config: ``tf.ConfigProto`` overrides.
    """
    self._model = model
    self._config = config
    self._num_devices = num_devices

    # lemmatizer variables
    self.f_input=[]
    self.f_output=io.StringIO()
    self.localcache={} #tokendata -> lemma

    session_config_base = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(
            allow_growth=gpu_allow_growth))

    # Disable layout optimizer for better conv1d performance, see:
    # https://github.com/tensorflow/tensorflow/issues/20309
    # This field does not exist in TensorFlow 1.4, so guard against the
    # exception.
    try:
      rewrite_options = text_format.Parse("""
          graph_options {
            rewrite_options {
              layout_optimizer: OFF
            }
          }
          """, tf.ConfigProto())
      session_config_base.MergeFrom(rewrite_options)
    except text_format.ParseError:
      pass

    if session_config is not None:
      session_config_base.MergeFrom(session_config)
    session_config = session_config_base
    run_config = tf.estimator.RunConfig(
        model_dir=self._config["model_dir"],
        session_config=session_config,
        tf_random_seed=seed)

    # Create a first session to enforce GPU options.
    # See https://github.com/OpenNMT/OpenNMT-tf/issues/80.
    _ = tf.Session(config=session_config)

    np.random.seed(seed)
    random.seed(seed)

    if "train" in self._config:
      if "save_summary_steps" in self._config["train"]:
        run_config = run_config.replace(
            save_summary_steps=self._config["train"]["save_summary_steps"],
            log_step_count_steps=self._config["train"]["save_summary_steps"])
      if "save_checkpoints_steps" in self._config["train"]:
        run_config = run_config.replace(
            save_checkpoints_secs=None,
            save_checkpoints_steps=self._config["train"]["save_checkpoints_steps"])
      if "keep_checkpoint_max" in self._config["train"]:
        run_config = run_config.replace(
            keep_checkpoint_max=self._config["train"]["keep_checkpoint_max"])

    self._estimator = tf.estimator.Estimator(
        self._model.model_fn(num_devices=self._num_devices),
        config=run_config,
        params=self._config["params"])




  def infer(self,
            features_file,
            predictions_file=None,
            checkpoint_path=None,
            log_time=False):
    """Runs inference.

    Args:
      features_file: The file(s) to infer from.
      predictions_file: If set, predictions are saved in this file.
      checkpoint_path: Path of a specific checkpoint to predict. If ``None``,
        the latest is used.
      log_time: If ``True``, several time metrics will be printed in the logs at
        the end of the inference loop.
    """
    if "infer" not in self._config:
      self._config["infer"] = {}
    if checkpoint_path is not None and os.path.isdir(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    batch_size = self._config["infer"].get("batch_size", 1)
    input_fn = self._model.input_fn(
        tf.estimator.ModeKeys.PREDICT,
        batch_size,
        self._config["data"],
        features_file,
        num_threads=self._config["infer"].get("num_threads"),
        prefetch_buffer_size=self._config["infer"].get("prefetch_buffer_size"))

    if predictions_file:
      stream = predictions_file
    else:
      stream = sys.stdout

    infer_hooks = []
    if log_time:
      infer_hooks.append(hooks.LogPredictionTimeHook())

    for prediction in self._estimator.predict(
        input_fn=input_fn,
        checkpoint_path=checkpoint_path,
        hooks=infer_hooks):
      self._model.print_prediction(prediction, params=self._config["infer"], stream=stream)

#    if predictions_file:
#      stream.close()


  def infer_batch(self, data_batch):

    """ Lemmatize one data batch """

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
            if token[LEMMA]!="_": # already filled in for example by another module, do not process
                continue
            token_data=(token[FORM],token[UPOS],token[XPOS],token[FEAT])
            if token_data not in self.localcache and token_data not in submitted:
                submitted.add(token_data)
                submitted_tdata.append(token_data)
                form, _ = transform_token(token)
                self.f_input.append(form) #print(form, file=self.f_input)
    #self.f_input.flush()
    print(" >>> {}/{} unique tokens submitted to lemmatizer".format(len(submitted_tdata),token_counter),file=sys.stderr)
    # run lemmatizer if everything is not in cache
    if len(submitted_tdata)>0:
        #self.f_input.seek(0) # beginning of the virtual file

        #with open("tmp.tmp", "wt", encoding="utf-8") as f:
        #    for t in self.f_input:
        #        print(t,file=f)

        np_input=np.array([t.encode("utf-8") for t in self.f_input])

        self.infer(np_input, self.f_output)

        #with open("tmp.tmp", "wt", encoding="utf-8") as f:
        #    for l in self.f_input:
        #        print(l,file=f)
        #self.infer("tmp.tmp", self.f_output)

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

    self.f_input=[] #self.f_input=io.StringIO() # clear virtual files
    self.f_output=io.StringIO()
#    self.translator.out_file=self.f_output

    return "\n".join(output_lines)+"\n"



def main(args):

    # init and load models
    yaml_config=load_config([args.config])
    lemmatizer_model=load_model(yaml_config["model_dir"], model_name="NMTSmall") # is model name part of config?
    print("model loaded")
    lemmatizer=Lemmatizer(lemmatizer_model, yaml_config)

    # input file
    if args.input_file:
        corpus_file = open(args.input_file, "rt", encoding="utf-8")
    else:
        corpus_file = sys.stdin

    # output file
    if args.output_file:
        real_output_file=open(args.output_file, "wt", encoding="utf-8")
    else:
        real_output_file=sys.stdout


    # lemmatize
    for batch in nonblocking_batches(f=corpus_file):

        lemmatized_batch=lemmatizer.infer_batch(batch)
        print(lemmatized_batch, file=real_output_file, flush=True, end="")


    # close files if needed
    if args.input_file:
        corpus_file.close()
    if args.output_file:
        real_output_file.close()



if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description='Lemmatize conllu text')
    argparser.add_argument('--config', type=str, help='Config file')
    argparser.add_argument('--input_file', type=str, default="", help='Input file')
    argparser.add_argument('--output_file', type=str, default="", help='Output file')
    args = argparser.parse_args()

    main(args)
