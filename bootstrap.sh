#!/bin/sh

DATA=`/usr/bin/env python3 -c "print(__import__('ppp_nlp_ml_standalone.config').config.get_data('embeddings-scaled.EMBEDDING_SIZE=25.txt.gz'))"`
wget http://metaoptimize.s3.amazonaws.com/cw-embeddings-ACL2010/embeddings-scaled.EMBEDDING_SIZE=25.txt.gz;
gzip -d embeddings-scaled.EMBEDDING_SIZE=25.txt.gz
mv embeddings-scaled.EMBEDDING_SIZE=25.txt $DATA
PYTHONPATH=$PWD ./demo/Dataset.py
PYTHONPATH=$PWD ./demo/Learn.py
