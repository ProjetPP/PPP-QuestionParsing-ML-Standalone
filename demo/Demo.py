#!/usr/bin/env python3
from ppp_questionparsing_ml_standalone import extract_triple

if __name__ == "__main__":
    extractTriple = extract_triple.ExtractTriple()
    lua = False
    while True:
        s = input()
        if s is not '':
            extractTriple.change_method("PythonLinear")
            extractTriple.print_triplet(extractTriple.extract_from_sentence(s))
            if lua:
                extractTriple.change_method("LuaLinear")
                extractTriple.print_triplet(extractTriple.extract_from_sentence(s))
