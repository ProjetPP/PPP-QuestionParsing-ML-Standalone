import os
import numpy

from . import dataset, linear_classifier, config, preprocessing


class TripleExtractor:
    __dictionary = None
    __linear_predict = None
    __fs = None
    __method = ""

    def __init__(self):
        self.__dictionary = preprocessing.Dictionary(config.get_data('embeddings-scaled.EMBEDDING_SIZE=25.txt'))
        p = linear_classifier.Predict()
        self.__linear_predict = p

    def extract_from_sentence(self, sentence):
        self.__fs = dataset.FormatSentence(sentence, self.__dictionary, window_size=config.get_windows_size())

        input_matrix = self.__fs.data_set_input()
        output_matrix = self.__linear_predict.predict(input_matrix)
        return self.get_triplet(numpy.argmax(output_matrix, axis=1))

    def get_triplet(self, solution):
        a, b, c = [], [], []

        for i in range(0, solution.shape[0]):
            if int(solution[i]) == 0:
                a.append(self.__fs.words[i])
            elif int(solution[i]) == 1:
                b.append(self.__fs.words[i])
            elif int(solution[i]) == 2:
                c.append(self.__fs.words[i])

        def get_elem(l):
            if len(l) == 0:
                return '?'
            else:
                return ' '.join(l)

        return get_elem(a), get_elem(b), get_elem(c)

    @staticmethod
    def print_triplet(triplet):
        print("(%s, %s, %s)" % triplet)