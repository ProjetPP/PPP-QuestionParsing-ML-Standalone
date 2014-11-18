from ppp_nlp_ml_standalone import ExtractTriple, config

file = config.get_data('trec1999.txt')
f = open(file, 'r')

extractTriplet = ExtractTriple.ExtractTriple()


for sentence in f:
    sentence = sentence[:-1]
    print(sentence)
    extractTriplet.print_triplet(extractTriplet.extract_from_sentence(sentence))
    print('')