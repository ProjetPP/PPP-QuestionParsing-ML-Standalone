import ExtractTriplet

file = '../data/trec1999.txt'
f = open(file, 'r')

extractTriplet = ExtractTriplet.ExtractTriplet()


for sentence in f:
    sentence = sentence[:-1]
    print(sentence)
    extractTriplet.extract_from_sentence(sentence)
    print('')