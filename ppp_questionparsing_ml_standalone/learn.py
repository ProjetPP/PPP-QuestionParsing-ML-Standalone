from . import linear_classifier, config

def learn():
    """Function called while bootstrapping."""
    trainModel = linear_classifier.TrainLinearClassifier(config.get_data('questions.train.txt'),
                                                         config.get_data('answers.train.txt'))
    trainModel.train(n_epochs=5500, learning_rate=0.001, l2_reg=0.001)
    trainModel.train_evaluation()
    trainModel.test_evaluation(config.get_data('questions.test.txt'),
                               config.get_data('answers.test.txt'))
    trainModel.save_model()


