import os
import numpy

from . import dataset, linear_classifier, config


class TripleExtractor:
    __dictionary = None
    __linear_predict = None
    __fs = None
    __method = ""

    def __init__(self, method="PythonLinear"):
        self.__dictionary = dataset.Dictionary(config.get_data('embeddings-scaled.EMBEDDING_SIZE=25.txt'))
        p = linear_classifier.Predict()
        self.__linear_predict = p
        self.__method = method

    def change_method(self, method):
        self.__method = method

    def extract_from_sentence(self, sentence):
        self.__fs = dataset.FormatSentence(sentence, self.__dictionary, window_size=5)

        if self.__method == "PythonLinear":
            input_matrix = self.__fs.numpy_input()
            output_matrix = self.__linear_predict.predict(input_matrix)
            return self.get_triplet(numpy.argmax(output_matrix, axis=1))

        elif self.__method == "LuaLinear":
            fs = dataset.FormatSentence(sentence, self.__dictionary)
            file = open(config.get_config_path() + 'input.txt', 'w')
            file.write(fs.data_set_input())
            file.close()

            os.system('cd ' + config.get_data('../ppp_ml_lua; th forward.lua'))
            result = open(config.get_data('output.txt'), 'r')

            return self.get_triplet(numpy.array(list(map(lambda x: int(x) - 1, result))))

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


    @staticmethod
    def is_yes_no_question(first_word):
        list_fw = ['is', 'are', 'am', 'was', 'were', 'will', 'do', 'does', 'did', 'have', 'had', 'has', 'can', 'could',
                   'should', 'shall', 'may', 'might', 'would']

        return first_word in list_fw

        #Compute all the possibilities of assignation of each word, and then choose the better solution according to
        # some rules
        # But it is too slow in python...Need to be wrapped in C++ or in C.

        # def coherent_triplet(self, output_matrix, fs):
        #
        #     print(numpy.prod(numpy.max(output_matrix, axis=1)))
        #     number_words = len(fs.words)
        #     number_of_holes = 1
        #     if self.is_yes_no_question(fs.words[0]):
        #         number_of_holes = 0
        #
        #     eye4 = numpy.eye(4, dtype=float)
        #
        #     #retourne une matrice maximisant le produit des proba
        #
        #     def test_possibilities(vector, i):
        #         if i == number_words:
        #             #print(vector)
        #             sol = numpy.sum(output_matrix * vector, axis=1)
        #
        #             return numpy.prod(sol)
        #         else:
        #             #On choisit une catégorie pour le mot i parmis 4 possible:
        #             max_prob = -1
        #
        #             for j in range(0, 4):
        #                 vector[i] = eye4[j]
        #                 prob = test_possibilities(vector, i+1)
        #                 if prob > max_prob:
        #                     max_prob = prob
        #
        #             return max_prob
        #
        #     return test_possibilities(numpy.zeros((number_words, 4), dtype=float), 0)

