#Goal

The goal of dataset.py is to transform English questions in a vectorized form that is compatible with the
neuron network.


## Requirements

You need to install nltk and to download files in order that nltk.word_tokenize works.

    sudo pip3 install nltk

In a python shell:

    nltk.download()

You need also to download the dictionary of vectors, here: http://metaoptimize.com/projects/wordreprs/

    wget http://metaoptimize.s3.amazonaws.com/cw-embeddings-ACL2010/embeddings-scaled.EMBEDDING_SIZE=200.txt.gz
    tar -xvf embeddings-scaled.EMBEDDING_SIZE=200.txt.gz

## Usage

The file AnnotatedQuestions.txt contains the questions in English, and the triplets associated to the questions.

Run python3 dataset.py to build the dataset.
The script will create two files: questions.txt and answers.txt

