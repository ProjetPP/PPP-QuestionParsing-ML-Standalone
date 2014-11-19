"""
 Logistic Regression

 References :
   - Jason Rennie: Logistic Regression,
   http://qwone.com/~jason/writing/lr.pdf

"""

import numpy

from . import utils, config


class LogisticRegression(object):
    def __init__(self, input_matrix, label, n_in, n_out):
        self.x = input_matrix
        self.y = label
        self.W = numpy.zeros((n_in, n_out))  # initialize W 0
        self.b = numpy.zeros(n_out)          # initialize bias 0

        # self.params = [self.W, self.b]

    def train(self, lr=0.1, input_matrix=None, L2_reg=0.00):
        if input_matrix is not None:
            self.x = input_matrix

        # p_y_given_x = sigmoid(numpy.dot(self.x, self.W) + self.b)
        p_y_given_x = utils.softmax(numpy.dot(self.x, self.W) + self.b)
        d_y = self.y - p_y_given_x

        self.W += lr * numpy.dot(self.x.T, d_y) - lr * L2_reg * self.W
        self.b += lr * numpy.mean(d_y, axis=0)

        # cost = self.negative_log_likelihood()
        # return cost

    def negative_log_likelihood(self):
        # sigmoid_activation = sigmoid(numpy.dot(self.x, self.W) + self.b)
        sigmoid_activation = utils.softmax(numpy.dot(self.x, self.W) + self.b)

        cross_entropy = - numpy.mean(
            numpy.sum(self.y * numpy.log(sigmoid_activation) +
                      (1 - self.y) * numpy.log(1 - sigmoid_activation), axis=1))

        return cross_entropy

    def predict(self, x):
        # return sigmoid(numpy.dot(x, self.W) + self.b)
        return utils.softmax(numpy.dot(x, self.W) + self.b)


class TrainLinearClassifier:
    train_in = None
    train_out = None
    vector_out = None

    n_in, n_out, m = 0, 0, 0

    test_in = None
    test_out = None

    classifier = None

    __debug = True

    def __init__(self, file_train_in, file_train_out, debug=True):
        self.__build_x(file_train_in)
        self.__build_y(file_train_out)

        self.classifier = LogisticRegression(input_matrix=self.train_in, label=self.train_out, n_in=self.n_in,
                                             n_out=self.n_out)
        self.__debug = debug
        if self.__debug:
            print('Size of a vector: %d' % self.n_in)

    def train(self, n_epochs=1500, learning_rate=0.001, l2_reg=0.001):
        for epoch in range(n_epochs):
            self.classifier.train(lr=learning_rate, L2_reg=l2_reg)
            cost = self.classifier.negative_log_likelihood()
            if epoch % 200 == 0 and self.__debug:
                print('Training epoch %d, cost is ' % epoch, cost)


    def train_evaluation(self):
        #from sklearn.metrics import confusion_matrix
        #Eval on the training set
        estimated_answers_vector = 1+numpy.argmax(self.classifier.predict(self.train_in), axis=1)
        answers_vector = self.vector_out

        correct_answers = numpy.sum((estimated_answers_vector == answers_vector).astype('int'))
        ratio_correct_answers = float(correct_answers)/answers_vector.shape[0]
        if self.__debug:
            print('Ratio of correct answers on the training set: %f' % ratio_correct_answers)
        return ratio_correct_answers
        #print('Confusion Matrix: ')
        #print(confusion_matrix(estimated_answers_vector, answers_vector))

    def save_model(self):
        numpy.save(config.get_data('W.npy'), self.classifier.W)
        numpy.save(config.get_data('b.npy'), self.classifier.b)

    def __build_x(self, file_in):
        self.train_in = numpy.loadtxt(file_in)
        self.n_in = self.train_in.shape[1]
        self.m = self.train_in.shape[0]

    def __build_y(self, file_in):
        vector = numpy.loadtxt(file_in)
        self.n_out = numpy.max(vector)
        self.vector_out = vector
        m = vector.shape[0]
        self.train_out = numpy.fromfunction(lambda i, j: vector[i] == j+1, (m, self.n_out), dtype=int).astype('int')

    def test_evaluation(self, file_test_in, file_test_out):
        #from sklearn.metrics import confusion_matrix
        self.test_in = numpy.loadtxt(file_test_in)
        answers_vector = numpy.loadtxt(file_test_out)
        estimated_answers_vector = 1+numpy.argmax(self.classifier.predict(self.test_in), axis=1)

        correct_answers = numpy.sum((estimated_answers_vector == answers_vector).astype('int'))
        ratio_correct_answers = float(correct_answers)/answers_vector.shape[0]
        if self.__debug:
            print('Ratio of correct answers on the test set: %f' % ratio_correct_answers)

        return ratio_correct_answers
        #print('Confusion Matrix: ')
        #print(confusion_matrix(estimated_answers_vector, answers_vector))


class Predict:
    W = None
    b = None

    def __init__(self):
        self.W = numpy.load(config.get_data('W.npy'))
        self.b = numpy.load(config.get_data('b.npy'))

    def predict(self, input_matrix):
        return utils.softmax(numpy.dot(input_matrix, self.W) + self.b)