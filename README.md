#Goal of PPP-NLP-ML-standalone

The goal is to understand the semantic of an English question.
We provide here a tool to transform an English question into a triplet:
(subject, predicate, object)

You can find some examples of this transformation is the file data/AnnotatedQuestions.txt.

##Generate the data set

The goal of ppp_nlp_ml_standalone/Dataset.py is to transform English questions in a vectorized form that is compatible with the
neuron network, according to a lookup table.

-After cloning this repo, the first thing to do is to download the lookup table here:

    cd data
    wget http://metaoptimize.s3.amazonaws.com/cw-embeddings-ACL2010/embeddings-scaled.EMBEDDING_SIZE=25.txt.gz
    tar -xvf embeddings-scaled.EMBEDDING_SIZE=25.txt

-You need to install nltk and to download some dependencies files in order that nltk.word_tokenize works:

    sudo pip3 install nltk

In a python shell:

    nltk.download()

-The english data set of questions is in the file: data/AnnotatedQuestions.txt
Compile the data set with the command:

    cd ppp_nlp_ml_standalone
    python3 Dataset.py

##Learn the Python model

You need numpy (pip3 install numpy):

    cd ppp_nlp_ml_standalone
    python3 LinearClassifier

##Learn the Torch7 model (this is optional)

You need torch7: http://torch.ch/
After installed it, you can execute the following command to learn the parameters:

    cd ppp_ml_lua; th -i neuron-network.lua
    for i = 1,100 do train() end
    test()


##Use the tool

Execute the command:

    cd ppp_nlp_ml_standalone
    python3 extractTriplet.py

Type a question in English, and the program will compute the triplet associated to the question.
Example:

    Is Dustin Hoffman an actor?
    (Dustin Hoffman, Is, an actor)

