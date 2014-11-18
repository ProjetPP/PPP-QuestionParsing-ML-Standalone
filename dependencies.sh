#!/bin/sh

cd data;
wget http://metaoptimize.s3.amazonaws.com/cw-embeddings-ACL2010/embeddings-scaled.EMBEDDING_SIZE=25.txt.gz;
gzip -d embeddings-scaled.EMBEDDING_SIZE=25.txt.gz;

