"""Request handler of the module."""

import ppp_datamodel
from ppp_datamodel import Sentence, Missing, Resource
from ppp_datamodel.communication import TraceItem, Response
from . import extract_triple


def missing_or_resource(x):
    return Missing() if x == '?' else Resource(value=x)


class RequestHandler:
    def __init__(self, request):
        self.request = request

    def answer(self):
        if not isinstance(self.request.tree, Sentence):
            return []

        sentence = self.request.tree.value
        extract_triplet = extract_triple.ExtractTriple()
        triple = extract_triplet.extract_from_sentence(sentence)
        (subject, predicate, object) = map(missing_or_resource, triple)

        triple = ppp_datamodel.Triple(subject=subject,
                                      predicate=predicate,
                                      object=object)

        meas = {'accuracy': 0.5, 'relevance': 0.5}
        trace = self.request.trace + [TraceItem('NLP-ML-standalone', triple, meas)]
        response = Response('en', triple, meas, trace)
        print(repr(response))
        return [response]
