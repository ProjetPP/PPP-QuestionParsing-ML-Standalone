#!/usr/bin/env python3
from ppp_questionparsing_ml_standalone import TripleExtractor

if __name__ == "__main__":
    extractTriple = TripleExtractor()
    while True:
        s = input()
        if s is not '':
            extractTriple.print_triplet(extractTriple.extract_from_sentence(s))