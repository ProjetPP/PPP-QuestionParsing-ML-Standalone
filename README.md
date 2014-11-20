#PPP-QuestionParsing-ML-standalone

[![Build Status](https://scrutinizer-ci.com/g/ProjetPP/PPP-NLP-ML-standalone/badges/build.png?b=master)](https://scrutinizer-ci.com/g/ProjetPP/PPP-NLP-ML-standalone/build-status/master)
[![Code Coverage](https://scrutinizer-ci.com/g/ProjetPP/PPP-NLP-ML-standalone/badges/coverage.png?b=master)](https://scrutinizer-ci.com/g/ProjetPP/PPP-NLP-ML-standalone/?branch=master)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/ProjetPP/PPP-NLP-ML-standalone/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/ProjetPP/PPP-NLP-ML-standalone/?branch=master)

The goal is to understand the semantic of an English question.

We provide here a tool to transform an English question into a triple:
(subject, predicate, object)

You can find some examples of this transformation is the file data/AnnotatedQuestions.txt.

## How to install

Download the git depo:

```
git clone https://github.com/ProjetPP/PPP-NLP-ML-standalone
cd PPP-NLP-ML-standalone
```


Then, install the script:

    python3 setup.py install


Use the `--user` option if you want to install it only for the current user.

## Bootstrap

Short version: run `./bootstrap.sh`

Detailed version:

###Download the look-up table:

```
cd data
wget http://metaoptimize.s3.amazonaws.com/cw-embeddings-ACL2010/embeddings-scaled.EMBEDDING_SIZE=25.txt.gz
gzip -d embeddings-scaled.EMBEDDING_SIZE=25.txt.gz
```

###Generate the data set

The goal of ppp_questionparsing_ml_standalone/Dataset.py is to transform English questions in a vectorized form that is compatible
with our ml model, according to a lookup table.

The english data set of questions is in the file: data/AnnotatedQuestions.txt
Compile the data set with the command:

    python3 demo/Dataset.py

###Learn the Python model

    python3 demo/Learn.py

###Learn the Torch7 model (this is optional)

You need torch7: http://torch.ch/
After installed it, you can execute the following command to learn the parameters:

    cd ppp_ml_lua; th -i neuron-network.lua
    for i = 1,100 do train() end
    test()


##Use the tool in CLI

Execute the command:

    python3 demo/Demo.py

Type a question in English, and the program will compute the triple associated to the question.
Example:

    Is Dustin Hoffman an actor?
    (Dustin Hoffman, Is, an actor)

##Use the tool with the server

    gunicorn ppp_questionparsing_ml_standalone:app -b 127.0.0.1:8080

In a python shell:

    import requests, json
    requests.post('http://localhost:8080/', data=json.dumps({'id':
    'foo', 'language': 'en', 'measures': {}, 'trace': [], 'tree': {'type':
    'sentence', 'value': 'What is the birth date of George Washington?'}})).json()


