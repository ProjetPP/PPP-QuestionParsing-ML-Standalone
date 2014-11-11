from ppp_nlp_ml_standalone import ExtractTriplet, config

file = config.get_config_path()+ 'trec1999.txt'
f = open(file, 'r')

extractTriplet = ExtractTriplet.ExtractTriplet()


for sentence in f:
    sentence = sentence[:-1]
    print(sentence)
    extractTriplet.extract_from_sentence(sentence)
    print('')