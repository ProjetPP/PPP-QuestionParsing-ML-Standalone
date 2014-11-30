#!/bin/sh

export PPP_ML_STANDALONE_CONFIG=config.json
DATA_DIR=`/usr/bin/env python3 -c "print(__import__('ppp_questionparsing_ml_standalone.config').config.Config().data_dir)"`

mkdir -p $DATA_DIR

if [ ! -f $DATA_DIR/embeddings-scaled.EMBEDDING_SIZE=25.txt ]
then
    if [ ! -f embeddings-scaled.EMBEDDING_SIZE=25.txt.gz ]
    then
        wget http://metaoptimize.s3.amazonaws.com/cw-embeddings-ACL2010/embeddings-scaled.EMBEDDING_SIZE=25.txt.gz -c
    fi
    gzip -d embeddings-scaled.EMBEDDING_SIZE=25.txt.gz
    mv -v embeddings-scaled.EMBEDDING_SIZE=25.txt $DATA_DIR
fi
python3 -m ppp_questionparsing_ml_standalone bootstrap
