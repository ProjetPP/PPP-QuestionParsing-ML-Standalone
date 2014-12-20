#PPP-QuestionParsing-ML-standalone

[![Build Status](https://travis-ci.org/ProjetPP/PPP-QuestionParsing-ML-Standalone.svg?branch=master)](https://travis-ci.org/ProjetPP/PPP-QuestionParsing-ML-Standalone)
[![Code Coverage](https://scrutinizer-ci.com/g/ProjetPP/PPP-QuestionParsing-ML-standalone/badges/coverage.png?b=master)](https://scrutinizer-ci.com/g/ProjetPP/PPP-QuestionParsing-ML-standalone/?branch=master)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/ProjetPP/PPP-QuestionParsing-ML-standalone/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/ProjetPP/PPP-QuestionParsing-ML-standalone/?branch=master)

We provide here a tool to transform an English question into a triple:
(subject, predicate, object)

We emphasis on keywords questions like "Barack Obama birth date?"

You can find some examples of this transformation is the file data/AnnotatedQuestions.txt.

## How to install

Download the git repository:

```
git clone https://github.com/ProjetPP/PPP-QuestionParsing-ML-standalone
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
with our ML model, according to a lookup table.

The english data set of questions is in the file: data/AnnotatedQuestions.txt
Compile the data set with the command:

    python3 demo/Dataset.py

###Learn the Python model

    python3 demo/Learn.py


##Use the tool in CLI

Execute the command:

    python3 demo/Demo.py

Type a question in English, and the program will compute the triple associated to the question.
Example:

    birth date Barack Obama?
    (Barack Obama, birth date, ?)

##Use the tool with the server

    gunicorn ppp_questionparsing_ml_standalone:app -b 127.0.0.1:8080

In a python shell:

    import requests, json
    requests.post('http://localhost:8080/', data=json.dumps({'id':
    'foo', 'language': 'en', 'measures': {}, 'trace': [], 'tree': {'type':
    'sentence', 'value': 'What is the birth date of George Washington?'}})).json()


