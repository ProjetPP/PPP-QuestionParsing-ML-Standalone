"""
 Logistic Regression

 References :
   - Jason Rennie: Logistic Regression,
   http://qwone.com/~jason/writing/lr.pdf

"""

import numpy

from . import config
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix


class LinearDiscriminantAnalysis(object):
    def __init__(self, input_matrix, labels):
        self.x = input_matrix
        self.y = labels
        self.clf = LDA()

    def train(self):
        self.clf.fit(self.x, self.y)

    def predict(self, x):
        return self.clf.predict_proba(x)

    def save_model(self, file):
        joblib.dump(self.clf, file)


class Classifier:
    train_in = None
    train_out = None

    n_in, n_out, m = 0, 0, 0

    classifier = None

    __debug = True

    def __init__(self, file_train_in, file_train_out, debug=True):

        self.train_in = numpy.load(file_train_in)
        self.n_in = self.train_in.shape[1]
        self.m = self.train_in.shape[0]

        self.train_out = numpy.load(file_train_out)

        self.classifier = LinearDiscriminantAnalysis(input_matrix=self.train_in, labels=self.train_out)

        self.__debug = debug
        if self.__debug:
            print('Size of a vector: %d' % self.n_in)

    def train(self):
        self.classifier.train()

    def train_evaluation(self):
        estimated_answers_vector = 1+numpy.argmax(self.classifier.predict(self.train_in), axis=1)
        answers_vector = self.train_out

        correct_answers = numpy.sum((estimated_answers_vector == answers_vector).astype('int'))
        ratio_correct_answers = float(correct_answers)/answers_vector.shape[0]
        if self.__debug:
            print('Ratio of correct answers on the training set: %f' % ratio_correct_answers)
            print('Confusion Matrix: ')
            print(confusion_matrix(estimated_answers_vector, answers_vector))

        return ratio_correct_answers

    def save_model(self):
        self.classifier.save_model(config.get_data('model.pkl'))

    def test_evaluation(self, file_test_in, file_test_out):
        test_in = numpy.load(file_test_in)
        answers_vector = numpy.load(file_test_out)
        estimated_answers_vector = 1+numpy.argmax(self.classifier.predict(test_in), axis=1)

        correct_answers = numpy.sum((estimated_answers_vector == answers_vector).astype('int'))
        ratio_correct_answers = float(correct_answers)/answers_vector.shape[0]
        if self.__debug:
            print('Ratio of correct answers on the test set: %f' % ratio_correct_answers)
            print('Confusion Matrix: ')
            print(confusion_matrix(estimated_answers_vector, answers_vector))

        return ratio_correct_answers


class Predict:
    W = None
    b = None

    def __init__(self):
        model_file = config.get_data('model.pkl')
        self.clf = joblib.load(model_file)

    def predict(self, input_matrix):
        return self.clf.predict_proba(input_matrix)