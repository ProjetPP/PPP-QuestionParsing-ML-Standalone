"""Request handler of the module."""

import ppp_datamodel
from ppp_datamodel import Sentence, Missing, Resource
from ppp_datamodel.communication import TraceItem, Response
from .triple_extractor import TripleExtractor


def missing_or_resource(x):
    return Missing() if x == '?' else Resource(value=x)

triple_extractor = None

class RequestHandler:
    def __init__(self, request):
        global triple_extractor
        if not triple_extractor:
            triple_extractor = TripleExtractor()
        self.request = request

    def answer(self):
        if not isinstance(self.request.tree, Sentence):
            return []

        sentence = self.request.tree.value
        triple = triple_extractor.extract_from_sentence(sentence)
        (subject, predicate, object) = map(missing_or_resource, triple)

        triple = ppp_datamodel.Triple(subject=subject,
                                      predicate=predicate,
                                      object=object)

        meas = {'accuracy': 0.5, 'relevance': 0.5}
        trace = self.request.trace + [TraceItem('NLP-ML-standalone', triple, meas)]
        response = Response('en', triple, meas, trace)
        print(repr(response))
        return [response]
