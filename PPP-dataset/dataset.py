import nltk
import random

#the number of words in the dictionary that we load
number_words = 20000
#the size of a vector which code a word

#You can get the dictionary here:
#http://metaoptimize.com/projects/wordreprs/


#We build the dictionary
def make_dictionary(dictionary):
    f = open(dictionary, 'r')
    english_dict = {}

    for i in range(1, number_words):
        line = f.readline()
        s = line.split(' ')
        word = s[0]
        vector = s[1:]
        vector_float = []
        for j in range(0, len(vector)):
            vector_float.append(float(vector[j]))

        english_dict[word] = vector_float

    f.close()

    return english_dict


class FormatSentence:
    """
    Take a sentence, annotated or not, and generate the vectors associated

    """

    words = []
    __null_vector = []
    __size_vector = 26
    __dictionary = []
    __vectorized_words = []
    __window_size = 3
    __annotated_sentence = ('', '', '')
    __is_annotated = False

    def __init__(self, raw_sentence, dictionary, annotated_sentence=('', '', '')):
        self.words = nltk.word_tokenize(raw_sentence)
        self.__null_vector = self.__vector_to_string(self, [0.0] * self.__size_vector)
        self.__dictionary = dictionary

        self.__vector_words()

        if annotated_sentence != ('', '', ''):
            self.__annotated_sentence = annotated_sentence
            self.__is_annotated = True


    @staticmethod
    def __vector_to_string(self, vector_w):
        s_out = ''
        for r in range(0, len(vector_w) - 1):
            s_out += "%4.4f " % vector_w[r]

        s_out += "%4.4f" % vector_w[len(vector_w)-1]

        return s_out

    @staticmethod
    def word_to_vector(self, word):
        if word in self.__dictionary:
            v = list(self.__dictionary[word])
        elif word.lower() in self.__dictionary:
            v = list(self.__dictionary[word.lower()])
        elif word.capitalize() in self.__dictionary:
            v = list(self.__dictionary[word.capitalize()])
        elif word.isdigit():
            v = list(self.__dictionary['1995'])
        else:
            v = list(self.__dictionary['*UNKNOWN*'])

        #We add one feature to know if the word start with an upper letter or not.
        if word.upper() == word:
            v.append(1.0)
        elif word[0].upper() == word[0]:
            v.append(-1.0)
        else:
            v.append(0)

        return v

    #Compute all the vector corresponding to the words of the sentence.
    def __vector_words(self):
        self.__vectorized_words = []
        for word in self.words:
            vector_w = self.word_to_vector(self, word)
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

    #return a vector of the output, if the sentence is annotated.
    def data_set_output(self):
        if self.__is_annotated:
            words_subject = nltk.word_tokenize(self.__annotated_sentence[0])
            words_predicate = nltk.word_tokenize(self.__annotated_sentence[1])
            words_object = nltk.word_tokenize(self.__annotated_sentence[2])

            output = ''
            for w in self.words:
                if w.lower() in list(map(lambda x: x.lower(), words_subject)):
                    output += '1\n'
                elif w.lower() in list(map(lambda x: x.lower(), words_object)):
                    output += '3\n'
                elif w.lower() in list(map(lambda x: x.lower(), words_predicate)):
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
    __dictionary = []
    __window_size = 3
    __file = None
    __number_lines = 0
    data_set_input = []
    data_set_output = []

    def __init__(self, dictionary, file):
        self.__dictionary = dictionary
        self.__number_lines = sum(1 for line in open(file))
        self.__file = open(file, 'r')

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
            fS = FormatSentence(sentence, self.__dictionary, a_sentence)

            self.data_set_input.append(fS.data_set_input())
            self.data_set_output.append(fS.data_set_output())

    def save(self, file_input, file_output):
        f_in_train = open('train.' + file_input, 'w')
        f_in_test = open('test.' + file_input, 'w')
        f_out_train = open('train.' + file_output, 'w')
        f_out_test = open('test.' + file_output, 'w')

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




if __name__ == '__main__':
    en_dict = make_dictionary('embeddings-scaled.EMBEDDING_SIZE=25.txt')
    data_set = BuildDataSet(en_dict, 'AnnotatedQuestions.txt')
    data_set.build()
    data_set.save('questions.txt', 'answers.txt')

    print('Database generated.')
    print('Number of entries in the train set: '+ str(sum(1 for line in open('train.answers.txt'))))
    print('Number of entries in the test set: '+ str(sum(1 for line in open('test.answers.txt'))))


    q = 'What is the first album of Led Zeppelin?'
    fs = FormatSentence(q, en_dict)
    print(fs.data_set_input())