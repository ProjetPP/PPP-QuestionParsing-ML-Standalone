


#PPP-NLP-ML-standalone

The goal is to understand the semantic of an English question.

We provide here a tool to transform an English question into a triple:
(subject, predicate, object)

You can find some examples of this transformation is the file data/AnnotatedQuestions.txt.

## How to install

With a recent version of pip:

```
pip3 install git+https://github.com/ProjetPP/PPP-NLP-ML-standalone
```

With an older one:

```
git clone https://github.com/ProjetPP/PPP-NLP-ML-standalone
cd PPP-NLP-classical
python3 setup.py install
```

Use the `--user` option if you want to install it only for the current user.


Then, set the global variable PPP_ML_STANDALONE_CONFIG:

    export PPP_ML_STANDALONE_CONFIG=config.json

##Generate the data set

The goal of ppp_nlp_ml_standalone/Dataset.py is to transform English questions in a vectorized form that is compatible
with our ml model, according to a lookup table.

Download the lookup table here:

    cd data
    wget http://metaoptimize.s3.amazonaws.com/cw-embeddings-ACL2010/embeddings-scaled.EMBEDDING_SIZE=25.txt.gz
    gzip -d embeddings-scaled.EMBEDDING_SIZE=25.txt.gz

-The english data set of questions is in the file: data/AnnotatedQuestions.txt
Compile the data set with the command:

    python3 ppp_nlp_ml_standalone/Dataset.py

##Learn the Python model

    python3 ppp_nlp_ml_standalone/Linearclassifier.py

##Learn the Torch7 model (this is optional)

You need torch7: http://torch.ch/
After installed it, you can execute the following command to learn the parameters:

    cd ppp_ml_lua; th -i neuron-network.lua
    for i = 1,100 do train() end
    test()


##Use the tool in CLI

Execute the command:

    python3 ppp_nlp_ml_standalone/extractTriplet.py

Type a question in English, and the program will compute the triple associated to the question.
Example:

    Is Dustin Hoffman an actor?
    (Dustin Hoffman, Is, an actor)

##Use the tool with the server

    gunicorn ppp_nlp_ml_standalone:app -b 127.0.0.1:8080

In a python shell:

    import requests, json
    requests.post('http://localhost:8080/', data=json.dumps({'id':
    'foo', 'language': 'en', 'measures': {}, 'trace': [], 'tree': {'type':
    'sentence', 'value': 'What is the birth date of George Washington?'}})).json()


