#!/usr/bin/env python3
from ppp_questionparsing_ml_standalone import TripleExtractor, config

file = config.get_data('trec1999.txt')
f = open(file, 'r')

extractTriplet = TripleExtractor.TripleExtractor()


for sentence in f:
    sentence = sentence[:-1]
    print(sentence)
    extractTriplet.print_triplet(extractTriplet.extract_from_sentence(sentence))
    print('')
