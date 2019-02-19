class NNConfig(object):
    def __init__(self, loss, optimizer, metrics, epochs, verbose=2):
        """
        :param loss: name of loss function that should be used
        :param optimizer: name of optimizer that should be used
        :param metrics: list of training metrics
        :param epochs: number of epochs to train for
        :param verbose: verbosity level for Keras training
        """
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.epochs = epochs
        self.verbose = verbose


class KMeansConfig(object):
    def __init__(self, init):
        """
        :param init: initialization k-means++ or random

        """
        self.init = init


class SumatraConfig(object):
    """
    Allows to specify the sumatra tags and run reason in the config file.
    """
    def __init__(self, run_tags=[], run_reason=""):
        """
        :param run_tags: list of sumatra tags
        :param run_reason: string to describe the run reason
        """
        self.run_tags = run_tags
        self.run_reason = run_reason
