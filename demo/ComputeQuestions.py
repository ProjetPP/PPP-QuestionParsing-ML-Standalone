#!/usr/bin/env python3
from ppp_questionparsing_ml_standalone import TripleExtractor, config

file = config.get_data('trec2000.txt')
f = open(file, 'r')

extractTriplet = TripleExtractor()


def get_elem(e):
    if e == '?':
        return '_'
    else:
        return e

for sentence in f:
    sentence = sentence[:-1]
    print(sentence)
    (a, b, c) = extractTriplet.extract_from_sentence(sentence)

    print('%s | %s | %s\n' % (get_elem(a), get_elem(b), get_elem(c)))