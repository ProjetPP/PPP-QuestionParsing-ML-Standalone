import nltk
import numpy
from nltk.tag import map_tag


class Dictionary:
    """
        You can get the dictionary here:
        http://metaoptimize.com/projects/wordreprs/
        Load the vectors of the lookup table in the dictionary english_dict
    """
    dict = {}
    __number_words = 20000
    size_vectors = 26

    def __init__(self, path='', number_words=20000, size_vectors=26):
        self.size_vectors = size_vectors
        self.__number_words = number_words

        f = open(path, 'r')
        for i in range(1, number_words):
            line = f.readline()
            s = line.split(' ')
            word = s[0]
            vector = s[1:]
            vector_float = []
            for j in range(0, len(vector)):
                vector_float.append(float(vector[j]))

            self.dict[word] = vector_float

        f.close()

    def word_to_vector(self, word):
        if word in self.dict:
            v = list(self.dict[word])
        elif word.lower() in self.dict:
            v = list(self.dict[word.lower()])
        elif word.capitalize() in self.dict:
            v = list(self.dict[word.capitalize()])
        elif word.isdigit():
            v = list(self.dict['1995'])
        else:
            #print('Warning: the word %s is not in the look up table' % word)
            v = list(self.dict['*UNKNOWN*'])

        #We add one feature to know if the word start with an upper letter or not.
        if word.upper() == word:
            v.append(1.0)
        elif word[0].upper() == word[0]:
            v.append(-1.0)
        else:
            v.append(0)

        return v


class PosTag:
    @staticmethod
    def compute_pos_tag(tokens):

        pos_tagged = nltk.pos_tag(tokens)
        simplified_tags = [map_tag('en-ptb', 'universal', tag) for word, tag in pos_tagged]
        lookup = {
            'VERB': 0,
            'NOUN': 1,
            'PRON': 2,
            'ADJ': 3,
            'ADV': 4,
            'ADP': 5,
            'CONJ': 6,
            'DET': 7,
            'NUM': 8,
            'PRT': 9,
            'X': 10
        }

        vector_output = []
        for word in simplified_tags:
            word_v = numpy.zeros(11)
            if word in lookup:
                word_v[lookup[word]] = 1

            vector_output.append(word_v.tolist())
        return vector_output


class PreProcessing():

    @staticmethod
    def tokenize(sentence):
        return PreProcessing.escape(sentence)

    @staticmethod
    def escape(sentence):
        list_sentence = []
        acc_letters = ''
        opened = False
        for i in range(0, len(sentence)):
            if sentence[i] == '"':
                list_sentence.append((opened, acc_letters))
                acc_letters = ''
                opened = not opened
            else:
                acc_letters += sentence[i]
        if not acc_letters == '':
            list_sentence.append((opened, acc_letters))

        result = []
        for (is_opened, sentence) in list_sentence:
            if not is_opened:
                result.extend(nltk.word_tokenize(sentence))
            else:
                result.append(sentence)
        return result
