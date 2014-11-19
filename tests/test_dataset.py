from unittest import TestCase

import os
from ppp_nlp_ml_standalone import Dataset, config


class DataSetTest(TestCase):

    def testPath(self):
        path = config.get_data('test')
        self.assertEqual(os.path.abspath(path), os.path.abspath('data/test'))

    def testSentence(self):
        en_dict = Dataset.Dictionary(config.get_data('embeddings-scaled.EMBEDDING_SIZE=25.txt'))
        w_size = 5
        self.assertEquals(len(en_dict.word_to_vector('Obama')), en_dict.size_vectors)
        self.assertEquals(len(en_dict.word_to_vector('53')), en_dict.size_vectors)

        sentence = 'What is the birth date of Obama?'

        fs = Dataset.FormatSentence(sentence, en_dict, window_size=w_size,
                                    annotated_sentence=('Obama', 'birth date', '_'))

        self.assertEquals(fs.words, ['What', 'is', 'the', 'birth', 'date', 'of', 'Obama'])
        self.assertEquals(len(fs.data_set_input_word(2).split(' ')), (w_size*2-1) * en_dict.size_vectors)

        self.assertEquals(fs.numpy_input().shape[1], (w_size*2-1) * en_dict.size_vectors)
        self.assertEquals(fs.numpy_input().shape[0], len(fs.words))

        l_answer = fs.data_set_output().split('\n')[:-1]

        self.assertEquals(len(l_answer), len(fs.words))

        for c in l_answer:
            self.assertIn(c, ['1', '2', '3', '4'])

    def testDataSet(self):
        w_size = 5
        en_dict = Dataset.Dictionary(config.get_data('embeddings-scaled.EMBEDDING_SIZE=25.txt'))

        data_set = Dataset.BuildDataSet(en_dict, config.get_data('AnnotatedQuestions.txt'), window_size=w_size)


        self.assertEquals(data_set.format_question('Who are you?'), 'who are you')
        self.assertEquals(data_set.format_question('Who are you'), 'who are you')
        self.assertEquals(data_set.format_question('Who are you.'), 'who are you')

        data_set.build()

        self.assertEquals(len(data_set.data_set_input), len(data_set.data_set_output))

        data_set.save(config.get_data('questions'), config.get_data('answers'))

        self.assertTrue(os.path.isfile(config.get_data('questions.train.txt')))
        self.assertTrue(os.path.isfile(config.get_data('questions.test.txt')))
        self.assertTrue(os.path.isfile(config.get_data('answers.test.txt')))
        self.assertTrue(os.path.isfile(config.get_data('answers.train.txt')))

        nb_lines_questions = str(sum(1 for line in open(config.get_data('questions.train.txt'))))
        nb_lines_answers = str(sum(1 for line in open(config.get_data('answers.train.txt'))))

        self.assertEquals(nb_lines_questions, nb_lines_answers)