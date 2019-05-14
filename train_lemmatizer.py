import yaml
import sys
import os
from random import shuffle
from artificial_training_data import create_data as create_art_data
from transducer_training_data import create_data as create_trans_data
from prepare_data import create_data as create_treebank_data
import glob

thisdir=os.path.dirname(os.path.realpath(__file__))

def create_training_data(config):
    # overall steps: artificial, transducer, treebank, mix, print

    print("Creating training data...", file=sys.stderr)
    data=[]
    # use artificial data?
    if config["basic"]!=True and "artificial" in config and config["artificial"]==True:
        if "artificial_vocab" not in config: # create vocab from training data if it's not given
            config["artificial_vocab"]=config["train"]
        data+=create_art_data(config["artificial_vocab"], config["artificial_size"], config["artificial_tag"])

    # use transducer data?
    if config["basic"]!=True and "transducer" in config and config["transducer"]==True:
        data+=create_trans_data(config["transducer_data"], config["transducer_word_freq"], config["train"], config["transducer_size"], config["transducer_tag"]) # transducer, word_freq, treebank_data, max_words, extra_tag
    # treebank data
    data+=create_treebank_data(config["train"])
    shuffle(data)
    model_dir=config["model_dir"]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        # clear directory
        files = glob.glob(os.path.join(model_dir,"*"))
        for f in files:
            print("Deleting file", f,file=sys.stderr)
            os.remove(f)
    #else:
        
    with open(os.path.join(model_dir,"train.input"), "wt") as input_file, open(os.path.join(model_dir,"train.output"), "wt") as output_file:
        for input_, output_ in data:
            print(input_, file=input_file)
            print(output_, file=output_file)
    print("Total of {x} examples in the training data.".format(x=len(data)), file=sys.stderr)
    # ready


def train(config, args):

    # overall steps: create training data, create devel data, preprocess data, train model, test on devel, test on test

    create_training_data(config)

    # devel data
    print("Creating development data...", file=sys.stderr)
    data=create_treebank_data(config["dev"])
    shuffle(data)
    model_dir=config["model_dir"]
    if not os.path.exists(os.path.dirname(model_dir)):
        os.makedirs(os.path.dirname(model_dir))
    with open(os.path.join(model_dir,"dev.input"), "wt") as input_file, open(os.path.join(model_dir,"dev.output"), "wt") as output_file:
        for input_, output_ in data:
            print(input_, file=input_file)
            print(output_, file=output_file)
    print("Total of {x} examples in the development data.".format(x=len(data)), file=sys.stderr)

    # preprocess data
    print("Preprocessing data...", file=sys.stderr)
    os.system("python3 {workdir}/OpenNMT-py/preprocess.py -train_src {train_input} -train_tgt {train_output} -valid_src {dev_input} -valid_tgt {dev_output} -save_data {model} {params}".format(workdir=thisdir, train_input=os.path.join(model_dir,"train.input"), train_output=os.path.join(model_dir,"train.output"), dev_input=os.path.join(model_dir,"dev.input"), dev_output=os.path.join(model_dir,"dev.output"), model=os.path.join(model_dir,"model"), params=config["preprocess_parameters"]))
    
    # train
    print("Training model...", file=sys.stderr)
    os.system("python3 {workdir}/OpenNMT-py/train.py -data {model} -save_model {model} {params}".format(workdir=thisdir, model=os.path.join(model_dir,"model"), params=config["train_parameters"]))

    print("Done. Models saved in {x}.".format(x=model_dir), file=sys.stderr)

if __name__=="__main__":
    import argparse
    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('--config', default="config.yaml", help='YAML with different configurations, Default: config.yaml')
    argparser.add_argument('--treebank', default="fi_tdt", help='Which configuration to read from the config, Default: %(default)s')
    args = argparser.parse_args()

    with open(args.config) as f:
        config=yaml.load(f)

    if args.treebank not in config:
        print(args.treebank,"not defined in", args.config, file=sys.stderr)
        sys.exit(1)

    train(config[args.treebank], args)
