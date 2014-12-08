from unittest import TestCase

from ppp_questionparsing_ml_standalone import preprocessing


class PreProcessingTest(TestCase):

    def test_tokenize(self):

        l_w = preprocessing.PreProcessing.tokenize('Where is "Barack Obama" house?')
        self.assertEquals(l_w, ['Where', 'is', 'Barack Obama', 'house', '?'])