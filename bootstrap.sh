#!/bin/sh

export PPP_ML_STANDALONE_CONFIG=config.json
DATA_DIR=`/usr/bin/env python3 -c "print(__import__('ppp_questionparsing_ml_standalone.config').config.Config().data_dir)"`
/usr/bin/env python3 -c "from sklearn.lda import LDA"

mkdir -p $DATA_DIR

if [ ! -f $DATA_DIR/embeddings-scaled.EMBEDDING_SIZE=25.txt ]
then
    wget http://metaoptimize.s3.amazonaws.com/cw-embeddings-ACL2010/embeddings-scaled.EMBEDDING_SIZE=25.txt.gz -c
    gzip -d embeddings-scaled.EMBEDDING_SIZE=25.txt.gz
    cp -v embeddings-scaled.EMBEDDING_SIZE=25.txt $DATA_DIR
fi
python3 -m ppp_questionparsing_ml_standalone bootstrap
