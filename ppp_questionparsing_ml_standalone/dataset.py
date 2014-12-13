import os
import re
import numpy
import itertools

from . import config, dataset_generation, preprocessing


class FormatSentence:
    """
    Take a sentence, annotated or not, and generate the vectors associated

    """
    sentence = ''
    words = []
    __size_vector = 26+11
    __dictionary = None
    __vectorized_words = []
    __window_size = 4
    __annotated_sentence = ('', '', '')
    __is_annotated = False
    __pos_tag_active = False

    def __init__(self, raw_sentence, dictionary, annotated_sentence=('', '', ''), window_size=4, pos_tag_active=True):
        if raw_sentence[-1] == '?' or raw_sentence[-1] == '.':
            self.sentence = raw_sentence[:-1]
        else:
            self.sentence = raw_sentence

        self.words = preprocessing.PreProcessing.tokenize(self.sentence)
        self.__dictionary = dictionary

        self.__vector_words()

        if annotated_sentence != ('', '', ''):
            self.__annotated_sentence = annotated_sentence
            self.__is_annotated = True

        self.__window_size = window_size

        if pos_tag_active:
            self.__size_vector = 26+11
            self.pos_tag = preprocessing.PosTag.compute_pos_tag(self.words)
        else:
            self.__size_vector = 26
        self.__pos_tag_active = pos_tag_active




    #Compute all the vector corresponding to the words of the sentence.
    def __vector_words(self):
        self.__vectorized_words = []
        for word in self.words:
            vector_w = self.__dictionary.word_to_vector(word)
            self.__vectorized_words.append(vector_w)

    #return the vector in a list format corresponding to a word with a fixed window size
    def data_set_input_word(self, word_index):
        res = []
        zeros = [0.0] * self.__size_vector
        for i in range(word_index-self.__window_size+1, word_index+self.__window_size):
            if i < 0:
                res.extend(zeros)
            elif i >= len(self.__vectorized_words):
                res.extend(zeros)
            else:
                res.extend(self.__vectorized_words[i])
                if self.__pos_tag_active:
                    res.extend(self.pos_tag[i])
        return res

    def data_set_input(self):
        res = []
        for i in range(0, len(self.words)):
            res.append(self.data_set_input_word(i))
        return res

    @staticmethod
    def __lower_list(l):
        return list(map(lambda x: x.lower(), l))

    @staticmethod
    def __occurrence_words(word):
        match_obj = re.match(r'(.*)/(\d)+', word, re.M|re.I)
        if match_obj:
            return True, match_obj.group(1), int(match_obj.group(2))
        else:
            return False, word, 1

    @staticmethod
    def compute_occurrences_sentence(words_sentence):
        words_occurrences = {}
        output = []

        for w in words_sentence:
            if w in words_occurrences:
                words_occurrences[w] += 1
            else:
                words_occurrences[w] = 1
            output.append((w, words_occurrences[w]))

        return output

    def data_set_output(self):
        if self.__is_annotated:
            words_subject = list(map(self.__occurrence_words,
                                     self.__lower_list(
                                         preprocessing.PreProcessing.tokenize(self.__annotated_sentence[0]))))

            words_predicate = list(map(self.__occurrence_words,
                                       self.__lower_list(
                                           preprocessing.PreProcessing.tokenize(self.__annotated_sentence[1]))))
            words_object = list(map(self.__occurrence_words,
                                    self.__lower_list(
                                        preprocessing.PreProcessing.tokenize(self.__annotated_sentence[2]))))

            words_sentence = self.__lower_list(self.words)
            words_occurrences = self.compute_occurrences_sentence(words_sentence)

            def check(l_words):
                if l_words != [(False, '_', 1)]:
                    for (is_annotated_word, word, pos) in l_words:
                        if (not is_annotated_word) and words_sentence.count(word) > 1:
                            print('Warning: the word "%s" as more than one occurrence in the sentence \n %s \n'
                                  % (word, self.sentence))

                        if not ((word, pos) in words_occurrences):
                            print('Warning: %s is not in the sentence %s' % (word, self.sentence))

            def check_unique(l_a, l_b):
                intersection = set(l_a).intersection(set(l_b))
                if len(intersection) > 0:
                    print('Warning: there exist one element that is present in more than one group in the '
                          'sentence \n %s', self.sentence)

            check(words_subject)
            check(words_predicate)
            check(words_object)

            check_unique(words_subject, words_predicate)
            check_unique(words_predicate, words_object)
            check_unique(words_subject, words_object)

            output = []
            for (w,n) in words_occurrences :
                if ((True, w, n) in words_subject) or ((False, w, n) in words_subject):
                    output.append(1)
                elif ((True, w, n) in words_object) or ((False, w, n) in words_object):
                    output.append(3)
                elif ((True, w, n) in words_predicate) or ((False, w, n) in words_predicate):
                    output.append(2)
                else:
                    output.append(4)

            return output
        else:
            return []


class BuildDataSet:
    """
    Build a data set from annotated questions and a dictionary of vectors
    """
    __dictionary = None
    __window_size = 4
    __file = None
    __number_sentences = 0
    __number_test_entries = 0
    __number_train_entries = 0
    data_set_input = []
    data_set_output = []

    __sentences = {}

    def __init__(self, dictionary, file, window_size=4, pos_tag_active=True):
        self.__dictionary = dictionary
        self.__file = open(file, 'r')
        self.__window_size = window_size
        self.__number_lines = sum(1 for line in open(file))
        self.__pos_tag_active = pos_tag_active

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
        self.__number_sentences += 1
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
            f_s = FormatSentence(sentence, self.__dictionary, a_sentence, window_size=self.__window_size,
                                 pos_tag_active=self.__pos_tag_active)
            sentence=self.format_question(sentence)
            self.addSentence(sentence,f_s)

    def save(self, file_input, file_output, training_set_distribution=0.9):
        number_training_example = int(training_set_distribution * self.__number_sentences)
        v_test = numpy.random.permutation(self.__number_sentences) > number_training_example - 1

        self.__number_train_entries = 0
        self.__number_test_entries = 0

        matrix_in_train = []
        matrix_in_test = []

        matrix_out_train = []
        matrix_out_test = []

        for i in range(0, self.__number_sentences):
            if v_test[i]:
                self.__number_test_entries += len(self.data_set_input[i])
                matrix_in_test.append(self.data_set_input[i])
                matrix_out_test.append(self.data_set_output[i])
            else:
                self.__number_train_entries += len(self.data_set_input[i])
                matrix_in_train.append(self.data_set_input[i])
                matrix_out_train.append(self.data_set_output[i])

        numpy.save(file_input + '.train.npy', numpy.concatenate(matrix_in_train))
        numpy.save(file_input + '.test.npy', numpy.concatenate(matrix_in_test))

        numpy.save(file_output + '.train.npy', numpy.concatenate(matrix_out_train))
        numpy.save(file_output + '.test.npy', numpy.concatenate(matrix_out_test))

    def number_train_entries(self):
        return self.__number_train_entries

    def number_test_entries(self):
        return self.__number_test_entries

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
            s = " ".join(sentence)
            f_s = FormatSentence(s, self.__dictionary, triple, self.__window_size, pos_tag_active=self.__pos_tag_active)
            self.addSentence(s,f_s)

    def generate_person(self):
        for p in dataset_generation.person:
            for ev in {"death", "birth"}:
                for obj in {"place", "date"}:
                    self.generateSentence(p, [obj, ev])


    def generate_country(self):
        for c in dataset_generation.country:
            self.generateSentence(c,["president", "capital"])
            self.generateSentence(c,["prime", "minister"])

    def generate_city(self):
        for c in dataset_generation.city:
            self.generateSentence(c,["mayor", "population", "country"])

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

    def generate_band(self):
        for g in dataset_generation.band:
            self.generateSentence(g, ["formation", "location"])
            self.generateSentence(g, ["formation", "date"])
            self.generateSentence(g, ["members"])

    def generate_all(self):
        self.generate_person()
        self.generate_country()
        self.generate_city()
        self.generate_location()
        self.generate_film()
        self.generate_book()
        self.generate_single()
        self.generate_art()


def create_dataset(training_set_distribution=0.95):
    """Function called when bootstraping to train the parser."""
    w_size = config.get_windows_size()

    en_dict = preprocessing.Dictionary(config.get_data('embeddings-scaled.EMBEDDING_SIZE=25.txt'))

    filename = os.path.join(os.path.dirname(__file__),
                            'data/AnnotatedQuestions.txt')
    data_set = BuildDataSet(en_dict, filename, window_size=w_size, pos_tag_active=True)
    data_set.build()
    data_set.generate_all()
    data_set.save(config.get_data('questions'), config.get_data('answers'),
                  training_set_distribution=training_set_distribution)

    print('Generated files saved in: \n' + config.get_data(''))

    print('Database generated.')
    print('Number of entries in the train set: ' + str(data_set.number_train_entries()))
    print('Number of entries in the test set: ' + str(data_set.number_test_entries()))