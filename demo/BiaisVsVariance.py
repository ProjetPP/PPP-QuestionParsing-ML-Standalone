#!/usr/bin/env python3
from ppp_questionparsing_ml_standalone import dataset, linear_classifier, config
import matplotlib.pyplot as plt
import os
import numpy


w_size = 5
en_dict = dataset.Dictionary(config.get_data('embeddings-scaled.EMBEDDING_SIZE=25.txt'))
filename = os.path.join('ppp_questionparsing_ml_standalone/data/AnnotatedQuestions.txt')
data_set = dataset.BuildDataSet(en_dict, filename, window_size=w_size)
data_set.build()
#data_set.generate_all()


training_set_distribution_list = numpy.arange(0.15, 0.9, 0.0025)

alpha = []
result_train = []
result_test = []

alpha_old = 0
for d in training_set_distribution_list:
    data_set.save(config.get_data('questions'), config.get_data('answers'),
                  training_set_distribution=d)

    train_set_size = sum(1 for line in open(config.get_data('questions.train.txt')))
    test_set_size = sum(1 for line in open(config.get_data('questions.test.txt')))

    alpha_new = train_set_size/(test_set_size+train_set_size)

    if alpha_new > alpha_old:

        trainModel = linear_classifier.TrainLinearClassifier(config.get_data('questions.train.txt'),
                                                             config.get_data('answers.train.txt'), debug=False)
        trainModel.train()

        train = 1-trainModel.train_evaluation()
        test = 1-trainModel.test_evaluation(config.get_data('questions.test.txt'),
                                          config.get_data('answers.test.txt'))

        print('%f : %f. Accuracy test: %f Accuracy learn: %f' % (d, alpha_new, 1-test, 1-train))

        result_test.append(test)
        result_train.append(train)
        alpha.append(alpha_new)
        alpha_old = alpha_new

plt.plot(alpha, result_train, 'ro')
plt.plot(alpha, result_test, '+')

plt.xlabel('Size of the learning set in %')
plt.ylabel('Error of the model')
plt.title('Biais vs Variance')
plt.ylim([0,0.5])
plt.savefig('BiaisVsVariance.png')
plt.show()
