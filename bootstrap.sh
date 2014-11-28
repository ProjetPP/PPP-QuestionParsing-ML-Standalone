#!/bin/sh

DATA_DIR=`/usr/bin/env python3 -c "print(__import__('ppp_questionparsing_ml_standalone.config').config.Config().data_dir)"`

mkdir -p $DATA_DIR

if [ ! -f $DATA_DIR/embeddings-scaled.EMBEDDING_SIZE=25.txt ]
then
    wget http://metaoptimize.s3.amazonaws.com/cw-embeddings-ACL2010/embeddings-scaled.EMBEDDING_SIZE=25.txt.gz -c
    gzip -d embeddings-scaled.EMBEDDING_SIZE=25.txt.gz
fi
mv -v embeddings-scaled.EMBEDDING_SIZE=25.txt $DATA_DIR
cp -v data/AnnotatedQuestions.txt $DATA_DIR
python3 -m ppp_questionparsing_ml_standalone bootstrap
