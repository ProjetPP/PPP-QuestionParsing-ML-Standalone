from . import linear_classifier, config


def learn():
    """Function called while bootstrapping."""
    trainModel = linear_classifier.Classifier(config.get_data('questions.train.npy'),
                                                         config.get_data('answers.train.npy'))
    trainModel.train()
    trainModel.train_evaluation()
    trainModel.test_evaluation(config.get_data('questions.test.npy'),
                               config.get_data('answers.test.npy'))
    trainModel.save_model()


