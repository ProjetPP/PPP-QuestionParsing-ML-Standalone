#!/usr/bin/env python3
from ppp_questionparsing_ml_standalone import preprocessing, dataset, linear_classifier, config
import os
import numpy


w_size = 5
en_dict = preprocessing.Dictionary(config.get_data('embeddings-scaled.EMBEDDING_SIZE=25.txt'))
filename = os.path.join('ppp_questionparsing_ml_standalone/data/AnnotatedQuestions.txt')
data_set = dataset.BuildDataSet(en_dict, filename, window_size=w_size, pos_tag_active=True)
data_set.build()
data_set.generate_all()

result_train = []
result_test = []

for x in range(0, 50):

    data_set.save(config.get_data('questions'), config.get_data('answers'),
                  training_set_distribution=0.9)

    train_set_size = data_set.number_train_entries()
    test_set_size = data_set.number_test_entries()

    alpha = train_set_size/(test_set_size+train_set_size)

    trainModel = linear_classifier.Classifier(config.get_data('questions.train.npy'),
                                              config.get_data('answers.train.npy'), debug=False)
    trainModel.train()

    train = trainModel.train_evaluation()
    test = trainModel.test_evaluation(config.get_data('questions.test.npy'),
                                      config.get_data('answers.test.npy'))

    print('%f - Accuracy test: %f - Accuracy learn: %f' % (alpha, test, train))

    result_test.append(test)
    result_train.append(train)


print('Mean test: %f, Mean train: %f, Std test: %f, Std train: %f' % (numpy.array(result_test).mean(),
                                                                      numpy.array(result_train).mean(),
                                                                      numpy.array(result_test).std(),
                                                                      numpy.array(result_train).std()))