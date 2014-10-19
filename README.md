#Goal of PPP-NLP-ML-sandalone

The goal is to understand the semantic of an English question.
We provide here a tool to transform an English question into a triplet.

Please find some examples of this transformation is the file PPP-dataset/AnnotatedQuestions.txt.

##How to use the tool

Execute the command:

    python3 extractTriplet.py

Type a question in English, and the program will compute the triplet associated to the question.
Example:

Is Dustin Hoffman an actor?
Dustin Hoffman | Is | an actor

##Requirements

You need python3 and the nltk module.
You also need torch7: http://torch.ch/

After cloning this repo, download the dictionary:

    cd PPP-dataset;
    wget http://metaoptimize.s3.amazonaws.com/cw-embeddings-ACL2010/embeddings-scaled.EMBEDDING_SIZE=200.txt.gz
    tar -xvf embeddings-scaled.EMBEDDING_SIZE=200.txt.gz

