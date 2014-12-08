import nltk


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
