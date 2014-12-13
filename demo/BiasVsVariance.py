#!/usr/bin/env python3
from ppp_questionparsing_ml_standalone import preprocessing, dataset, linear_classifier, config
import matplotlib.pyplot as plt
import os
import numpy


w_size = 5
en_dict = preprocessing.Dictionary(config.get_data('embeddings-scaled.EMBEDDING_SIZE=25.txt'))
filename = os.path.join('ppp_questionparsing_ml_standalone/data/AnnotatedQuestions.txt')
data_set = dataset.BuildDataSet(en_dict, filename, window_size=w_size, pos_tag_active=False)
data_set.build()
data_set.generate_all()


training_set_distribution_list = numpy.arange(0.15, 0.9, 0.0025)

alpha = []
result_train = []
result_test = []

alpha_old = 0
for d in training_set_distribution_list:
    data_set.save(config.get_data('questions'), config.get_data('answers'),
                  training_set_distribution=d)

    train_set_size = data_set.number_train_entries()
    test_set_size = data_set.number_test_entries()

    alpha_new = train_set_size/(test_set_size+train_set_size)

    if alpha_new > alpha_old:

        trainModel = linear_classifier.Classifier(config.get_data('questions.train.npy'),
                                                  config.get_data('answers.train.npy'), debug=False)
        trainModel.train()

        train = 1-trainModel.train_evaluation()
        test = 1-trainModel.test_evaluation(config.get_data('questions.test.npy'),
                                          config.get_data('answers.test.npy'))

        print('%f : %f. Accuracy test: %f Accuracy learn: %f' % (d, alpha_new, 1-test, 1-train))

        result_test.append(test)
        result_train.append(train)
        alpha.append(alpha_new)
        alpha_old = alpha_new

line_up, = plt.plot(alpha, result_train, 'ro', label='training set error')
line_down, = plt.plot(alpha, result_test, '+', label='testing set error')

plt.legend(handles=[line_up, line_down])

plt.xlabel('Size of the learning set in %')
plt.ylabel('Error of the model')
plt.ylim([0,0.5])
plt.savefig('BiasVsVariance.png')
#plt.show()
