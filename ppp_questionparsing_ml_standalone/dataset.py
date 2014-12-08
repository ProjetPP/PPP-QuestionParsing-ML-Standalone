import os
import nltk
import random
import numpy
import itertools

from . import config, dataset_generation, preprocessing


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
            v = list(self.dict['*UNKNOWN*'])

        #We add one feature to know if the word start with an upper letter or not.
        if word.upper() == word:
            v.append(1.0)
        elif word[0].upper() == word[0]:
            v.append(-1.0)
        else:
            v.append(0)

        return v


class FormatSentence:
    """
    Take a sentence, annotated or not, and generate the vectors associated

    """
    sentence = ''
    words = []
    __null_vector = []
    __size_vector = 26
    __dictionary = None
    __vectorized_words = []
    __window_size = 4
    __annotated_sentence = ('', '', '')
    __is_annotated = False

    def __init__(self, raw_sentence, dictionary, annotated_sentence=('', '', ''), window_size=4):
        if raw_sentence[-1] == '?' or raw_sentence[-1] == '.':
            self.sentence = raw_sentence[:-1]
        else:
            self.sentence = raw_sentence

        self.words = preprocessing.PreProcessing.tokenize(self.sentence)
        self.__null_vector = self.__vector_to_string(self, [0.0] * self.__size_vector)
        self.__dictionary = dictionary

        self.__vector_words()

        if annotated_sentence != ('', '', ''):
            self.__annotated_sentence = annotated_sentence
            self.__is_annotated = True

        self.__window_size = window_size


    @staticmethod
    def __vector_to_string(self, vector_w):
        s_out = ''
        for r in range(0, len(vector_w) - 1):
            s_out += "%4.4f " % vector_w[r]

        s_out += "%4.4f" % vector_w[len(vector_w)-1]

        return s_out

    #Compute all the vector corresponding to the words of the sentence.
    def __vector_words(self):
        self.__vectorized_words = []
        for word in self.words:
            vector_w = self.__dictionary.word_to_vector(word)
            self.__vectorized_words.append(self.__vector_to_string(self, vector_w))

    #return the vectors in a string format corresponding to a word with a fixed window size
    def data_set_input_word(self, word_index):
        res = ''
        for i in range(word_index-self.__window_size+1, word_index+self.__window_size):
            if i < 0:
                res += self.__null_vector
            elif i >= len(self.__vectorized_words):
                res += self.__null_vector
            else:
                res += self.__vectorized_words[i]

            if i < word_index+self.__window_size-1:
                res += ' '

        return res

    def data_set_input(self):
        res = ''
        for i in range(0, len(self.words)):
            res += self.data_set_input_word(i) + '\n'
        return res

    #Return a matrix corresponding to the input, in the numpy format
    def numpy_input(self):
        s = self.data_set_input().split('\n')
        matrix = numpy.zeros((len(self.words), (2*self.__window_size-1)*self.__size_vector), dtype=float)

        i = 0
        for line in s:
            if line is not '':
                v = numpy.fromstring(line, dtype=float, sep=' ')
                matrix[i] = v
                i += 1

        return matrix


    #return a vector of the output, if the sentence is annotated.

    @staticmethod
    def __lower_list(l):
        return list(map(lambda x: x.lower(), l))

    def data_set_output(self):
        if self.__is_annotated:
            words_subject = self.__lower_list(preprocessing.PreProcessing.tokenize(self.__annotated_sentence[0]))
            words_predicate = self.__lower_list(preprocessing.PreProcessing.tokenize(self.__annotated_sentence[1]))
            words_object = self.__lower_list(preprocessing.PreProcessing.tokenize(self.__annotated_sentence[2]))
            words_sentence = self.__lower_list(self.words)

            #print(words_subject)

            def check(l_words):
                if l_words != ['_']:
                    for w_l in l_words:
                        if not (w_l in words_sentence):
                            print('Warning: ' + w_l + ' is not in the sentence ' + self.sentence)
            check(words_subject)
            check(words_predicate)
            check(words_object)

            output = ''

            for w in self.words:
                if w.lower() in words_subject:
                    output += '1\n'
                elif w.lower() in words_object:
                    output += '3\n'
                elif w.lower() in words_predicate:
                    output += '2\n'
                else:
                    output += '4\n'
            return output
        else:
            return ''


class BuildDataSet:
    """
    Build a data set from annotated questions and a dictionary of vectors
    """
    __dictionary = None
    __window_size = 4
    __file = None
    __number_lines = 0
    data_set_input = []
    data_set_output = []

    __sentences = {}

    def __init__(self, dictionary, file, window_size=4):
        self.__dictionary = dictionary
        self.__number_lines = sum(1 for line in open(file))
        self.__file = open(file, 'r')
        self.__window_size = window_size

    def __del__(self):
        self.__file.close()

    @staticmethod
    def format_question(question):
        if question[-1] == '?' or question[-1] == '.':
            return question[:-1].lower()
        else:
            return question.lower()

    def addSentence(self,raw_sentence,format_sentence):
        self.data_set_input.append(format_sentence.data_set_input())
        self.data_set_output.append(format_sentence.data_set_output())
        if raw_sentence in self.__sentences:
            print('Warning: the sentence ' + raw_sentence + ' is already in the dataset')
        else:
            self.__sentences[raw_sentence] = True

    def build(self):
        for i in range(0, int((self.__number_lines+1)/3)):
            sentence = self.__file.readline()[:-1]

            s = self.__file.readline()[:-1].split('|')
            self.__file.readline()

            a, b, c = s[0], s[1], s[2]

            if a == '_':
                a = ''
            if b == '_':
                b = ''
            if c == '_':
                c = ''

            a_sentence = (a, b, c)
            f_s = FormatSentence(sentence, self.__dictionary, a_sentence, window_size=self.__window_size)
            sentence=self.format_question(sentence)
            self.addSentence(sentence,f_s)

    def save(self, file_input, file_output):
        f_in_train = open(file_input + '.train.txt', 'w')
        f_in_test = open(file_input + '.test.txt', 'w')
        f_out_train = open(file_output + '.train.txt', 'w')
        f_out_test = open(file_output + '.test.txt', 'w')

        for i in range(1, len(self.data_set_output)):
            if random.random() < 0.2:
                f_in_test.write(self.data_set_input[i])
                f_out_test.write(self.data_set_output[i])
            else:
                f_in_train.write(self.data_set_input[i])
                f_out_train.write(self.data_set_output[i])

        f_in_train.close()
        f_in_test.close()
        f_out_train.close()
        f_out_test.close()

    def generateSentence(self,subject,predicate):
        """
            Add all possible triples with a missing object, and the given subject and predicate.
            Subject must be a string.
            Predicate must be a list of string.
            self.generateSentence('foo',['bar1','bar2','bar3']) will generate all permutations of
                {'foo','bar1','bar2','bar3'}, associated to the triple ('foo', 'bar1 bar2 bar3', ?)
        """
        subject = subject.lower()
        predicate = [p.lower() for p in predicate]
        triple = (subject," ".join(predicate),"")
        for sentence in itertools.permutations(predicate+[subject]):
            f_s = FormatSentence(" ".join(sentence),self.__dictionary,triple,self.__window_size)
            self.addSentence(sentence,f_s)

    def generate_person(self):
        for p in dataset_generation.person:
            for ev in {"death","birth"}:
                for obj in {"place","date"}:
                    self.generateSentence(p,[obj,ev])

    def generate_country(self):
        for c in dataset_generation.country:
            self.generateSentence(c,["president"])
            self.generateSentence(c,["prime", "minister"])

    def generate_city(self):
        for c in dataset_generation.city:
            self.generateSentence(c,["mayor"])

    def generate_location(self):
        for l in dataset_generation.location:
            self.generateSentence(l,["population"])

    def generate_film(self):
        for f in dataset_generation.film:
            self.generateSentence(f,["cast","member"])
            self.generateSentence(f,["director"])

    def generate_book(self):
        for b in dataset_generation.book:
            self.generateSentence(b,["original","language"])
            self.generateSentence(b,["author"])

    def generate_single(self):
        for s in dataset_generation.single:
            self.generateSentence(s,["record","label"])

    def generate_art(self):
        for a in dataset_generation.art:
            self.generateSentence(a,["official","website"])
            self.generateSentence(a,["date","publication"])

    def generate_all(self):
        self.generate_person()
        self.generate_country()
        self.generate_city()
        self.generate_location()
        self.generate_film()
        self.generate_book()
        self.generate_single()
        self.generate_art()


def create_dataset():
    """Function called when bootstraping to train the parser."""
    w_size = 5

    en_dict = Dictionary(config.get_data('embeddings-scaled.EMBEDDING_SIZE=25.txt'))

    filename = os.path.join(os.path.dirname(__file__),
                            'data/AnnotatedQuestions.txt')
    data_set = BuildDataSet(en_dict, filename, window_size=w_size)
    data_set.build()
    data_set.generate_all()
    data_set.save(config.get_data('questions'), config.get_data('answers'))

    print('Generated files saved in: \n' + config.get_data(''))

    print('Database generated.')
    print('Number of entries in the train set: ' +
          str(sum(1 for line in open(config.get_data('questions.train.txt')))))
    print('Number of entries in the test set: ' +
          str(sum(1 for line in open(config.get_data('questions.test.txt')))))
