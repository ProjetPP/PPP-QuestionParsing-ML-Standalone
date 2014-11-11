import numpy


#numpy.seterr(all='ignore')

path_to_data = '/Users/quentin/Projet/PPP-NLP-ML-standalone/data/'

def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))


def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2
