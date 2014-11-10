"""Request handler of the module."""
import ppp_datamodel
from ppp_datamodel import Sentence
from ppp_datamodel.communication import TraceItem, Response
from ppp_core.exceptions import ClientError
from ppp_nlp_ml_standalone import ExtractTriplet


class RequestHandler:
    def __init__(self, request):
        self.request = request

    def answer(self):
        if not isinstance(self.request.tree, Sentence):
            return []

        sentence = self.request.tree.value
        extract_triplet = ExtractTriplet()
        a, b, c = extract_triplet.extract_from_sentence(sentence)

        if a == '?':
            subject = ppp_datamodel.Missing()
        else:
            subject = ppp_datamodel.Resource(value=a)

        if b == '?':
            predicate = ppp_datamodel.Missing()
        else:
            predicate = ppp_datamodel.Resource(value=b)

        if c == '?':
            object = ppp_datamodel.Missing()
        else:
            object = ppp_datamodel.Resource(value=b)

        triple = ppp_datamodel.Triple(subject=subject,
                                      predicate=predicate,
                                      object=object)

        meas = {'accuracy': 0.5, 'relevance': 0.5}
        trace = self.request.trace + [TraceItem('NLP-ML-standalone', triple, meas)]
        response = Response('en', triple, meas, trace)
        print(repr(response))
        return [response]