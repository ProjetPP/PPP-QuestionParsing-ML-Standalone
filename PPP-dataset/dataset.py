import nltk

#the number of words in the dictionary that we load
number_words = 2000
#the size of a vector which code a word


#We build the dictionary
def make_dictionary():
    f = open('embeddings-original.EMBEDDING_SIZE=200.txt', 'r')
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
    __size_vector = 201
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

        s_out += "%0.2f" % vector_w[len(vector_w)-1]

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
    def data_set_output_sentence(self):
        if self.__is_annotated:
            words_subject = nltk.word_tokenize(self.__annotated_sentence[0])
            words_predicate = nltk.word_tokenize(self.__annotated_sentence[1])
            words_object = nltk.word_tokenize(self.__annotated_sentence[2])

            output = ''
            for w in self.words:
                if w in words_subject:
                    output += '1\n'
                elif w in words_object:
                    output += '3\n'
                elif w in words_predicate:
                    output += '2\n'
                else:
                    output += '0\n'
            return output
        else:
            return ''


en_dict = make_dictionary()
sentence = "What's the birth date of Nicolas Sarkozy?"
a_sentence = ('Nicolas Sarkozy', 'birth date', '')

fS = FormatSentence(sentence, en_dict, a_sentence)
print(fS.data_set_input())
print(fS.data_set_output_sentence())
