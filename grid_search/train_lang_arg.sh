#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH -c 4


set -euxo pipefail

# TODO fill in data paths here!
TBDATA="" # path to directory where the treebank data directories are
WORKINGDIR="" # place to put result directories etc.

# argument reader (treebank, batch_size, dropout, epoch, gpuid, artificial data, copy attention)
# defaults:
BATCHSIZE=32
DROPOUT=0.01
EPOCHS=13
GPUID=0
ARTIFICIAL=0
COPYATTENTION=""
COPY="false"
DYNAMICDICT=""
for i in "$@"
do
case $i in
    -t=*|--treebank=*)
    TREEBANK="${i#*=}"
    shift # past argument=value
    ;;
    -b=*|--batchsize=*)
    BATCHSIZE="${i#*=}"
    shift # past argument=value
    ;;
    -d=*|--dropout=*)
    DROPOUT="${i#*=}"
    shift # past argument=value
    ;;
    -e=*|--epochs=*)
    EPOCHS="${i#*=}"
    shift # past argument=value
    ;;
    --gpuid=*)
    GPUID="${i#*=}"
    shift # past argument=value
    ;;
    --artificial_data=*)
    ARTIFICIAL="${i#*=}"
    shift # past argument with no value
    ;;
    --copy_attention=*)
    COPY="${i#*=}" # true or false
    if [ "$COPY" == "true" ] ; then
        COPYATTENTION="-copy_attn" # value must be -copy_attn
        DYNAMICDICT="-dynamic_dict"
    fi
    shift # past argument with no value
    ;;
    *)
          # unknown option
    ;;
esac
done

echo "treebank=$TREEBANK batch_size=$BATCHSIZE dropout=$DROPOUT epochs=$EPOCHS gpuid=$GPUID artificial_data=$ARTIFICIAL copy_atention=$COPY"

datadir=$WORKINGDIR/$TREEBANK/data
modeldir=$WORKINGDIR/$TREEBANK/model
griddir=$WORKINGDIR/$TREEBANK/grid

mkdir -p $datadir
mkdir -p $modeldir
mkdir -p $griddir
if [ -d "$modeldir" ]; then
    rm -f  $modeldir/*.pt
fi


## TRAINING DATA
python ../prepare_data.py -f $TBDATA/$TREEBANK/*train.conllu -o $datadir/train
python ../prepare_data.py -f $TBDATA/$TREEBANK/*dev.conllu -o $datadir/dev

train=train

## ARTIFICIAL TRAINING DATA
if ! [ "$ARTIFICIAL" -eq "0" ] ;
then
    python ../artificial_training_data.py -v ../universal_character_vocabulary -o $datadir/artificial.train --count $ARTIFICIAL --extra_tag t=ART

    # mix real and artificial data
    paste <(cat $datadir/artificial.train.input $datadir/train.input) <(cat $datadir/artificial.train.output $datadir/train.output) | shuf | tee >( cut -f 1 > $datadir/full.train.input) | cut -f 2 > $datadir/full.train.output

    train=full.train
fi

## TRAIN MODEL
python ../OpenNMT-py/preprocess.py -train_src $datadir/$train.input -train_tgt $datadir/$train.output -valid_src $datadir/dev.input -valid_tgt $datadir/dev.output -save_data $modeldir/lemmatizer $DYNAMICDICT

python ../OpenNMT-py/train.py -data $modeldir/lemmatizer -save_model $modeldir/lemmatizer-model -gpuid $GPUID -dropout $DROPOUT -batch_size $BATCHSIZE -epochs $EPOCHS -word_vec_size 200 $COPYATTENTION


# EVAL ON DEVEL
for epoch in 10 15 20 ; do
    outfile="tb=$TREEBANK|batchsize=$BATCHSIZE|dropout=$DROPOUT|epochs=$epoch|artificial=$ARTIFICIAL|copyattention=$COPY"
    python ../predict.py -model $modeldir/*e$epoch.pt -src $datadir/dev.udpipe.input -output $datadir/dev.predictions -gpu $GPUID
    cat $datadir/dev.output | perl -pe 's/ //g' | perl -pe 's/\$@@\$/ /g' > $datadir/dev.gold
    paste $datadir/dev.gold $datadir/dev.predictions > $griddir/$outfile.grid
done



