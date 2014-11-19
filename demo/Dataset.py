#!/usr/bin/env python3
from ppp_nlp_ml_standalone import Dataset, config

if __name__ == '__main__':
    w_size = 5

    en_dict = Dataset.Dictionary(config.get_data('embeddings-scaled.EMBEDDING_SIZE=25.txt'))

    data_set = Dataset.BuildDataSet(en_dict, config.get_data('AnnotatedQuestions.txt'), window_size=w_size)
    data_set.build()
    data_set.save(config.get_data('questions'), config.get_data('answers'))

    print('Generated files saved in: \n' + config.get_data(''))

    print('Database generated.')
    print('Number of entries in the train set: ' +
          str(sum(1 for line in open(config.get_data('questions.train.txt')))))
    print('Number of entries in the test set: ' +
          str(sum(1 for line in open(config.get_data('questions.test.txt')))))
