# universal-lemmatizer

Neural model for lemmatization using OpenNMT and pytorch libraries.

Needs python3 environment with pytorch installed.

## Prepering data from .conllu files

    python prepare_data.py -f train.conllu -o data/train
    python prepare_data.py -f devel.conllu -o data/devel
    
Creates data/train.input and data/train.output files.

