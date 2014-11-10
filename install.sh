#!/bin/bash

cd data;
wget http://metaoptimize.s3.amazonaws.com/cw-embeddings-ACL2010/embeddings-scaled.EMBEDDING_SIZE=25.txt.gz;
gzip -d embeddings-scaled.EMBEDDING_SIZE=25.txt.gz;
cd ../ppp_nlp_ml_standalone;
python3 Dataset.py;
python3 LinearClassifier.py;
