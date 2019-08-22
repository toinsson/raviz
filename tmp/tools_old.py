import time

import numpy as np
import pandas as pd

# openTSNE default
default_parameters = {
'n_components': 2,
'perplexity': 30,
'learning_rate': 200,
'early_exaggeration': 12,
'early_exaggeration_iter': 250,
'n_iter': 750,
'exaggeration': None,
'theta': 0.5,
'n_interpolation_points': 3,
'min_num_intervals': 50,
'ints_in_interval': 1,
'initialization': 'pca',
'metric': 'euclidean',
'metric_params': None,
'initial_momentum': 0.5,
'final_momentum': 0.8,
'min_grad_norm': 1e-08,
'max_grad_norm': None,
'n_jobs': 1,
'neighbors': 'approx',
'negative_gradient_method': 'fft',
'callbacks': None,
'callbacks_every_iters': 50,
'random_state': None
}

# own default
default_parameters['n_jobs'] = 8


class Callback(object):

    def __init__(self, logger):
        self.logger = logger
        self.data = np.zeros((0,3))

    def optimization_about_to_start(self):
        self.logger.info("starting")

    def __call__(self, iteration, error, embedding):
        self.logger.info("iter:{0: <8} KL_err:{1: <8}".format(iteration, error))
        self.data = np.vstack((self.data, [time.time(), iteration, error]))


import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def knn_classifier_performance(embedding, y, n_neighbors=1):
    split = sklearn.model_selection.ShuffleSplit(n_splits=1, random_state=0, test_size=0.25)
    id_train, id_test = next(split.split(embedding))
    Xemb_train = embedding[id_train]
    Xemb_test  = embedding[id_test]

    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(Xemb_train, y[id_train])
    y_pred = neigh.predict(Xemb_test)
    y_true = y[id_test]
    score = accuracy_score(y_true, y_pred)

    return score

def K_NN_classifier(data, labels, K=1):

    split = sklearn.model_selection.ShuffleSplit(n_splits=1, random_state=0, test_size=0.25)
    id_train, id_test = next(split.split(data))

    nnc_train = data[id_train]
    nnc_test  = data[id_test]

    neigh = KNeighborsClassifier(n_neighbors=K)
    neigh.fit(nnc_train, labels[id_train]) 

    y_pred = neigh.predict(nnc_test)
    y_true = labels[id_test]

    return accuracy_score(y_true, y_pred)



class CallbackKnn(object):

    def __init__(self, logger, y, freq=10):
        self.logger = logger
        self.y = y
        self.freq = 10

        self.data = np.zeros((0,4))

    def optimization_about_to_start(self):
        self.logger.info("starting")

    def __call__(self, iteration, error, embedding):

        if ((iteration % self.freq) == 0):
            score = knn_classifier_performance(embedding, self.y)
        else:
            score = -1

        self.logger.info("iter:{0: <8} KL_err:{1: <8} knn: {2:<8}".format(iteration, error, score))
        self.data = np.vstack((self.data, [time.time(), iteration, error, score]))





class ProfilingData(object):
    def __init__(self, name, params):
        self.params = params
        self.name   = name
        self.data_  = None

        # self.schema = None
        # self.dataset = None

    @property
    def data(self):
        return self.data_
    @data.setter
    def data(self, value):
        self.data_ = value



# load the dataset
def get_dataset(name):
    """from name return (X_train, y_train, X_test, y_test) if exist otherwise None for missing
    """
    # check the dataset name: mnist, mnist-fashion, macosko
    import sys

    if (name == "fashion-mnist"):
        sys.path.insert(0, "../research/fashion-mnist/")
        from utils import mnist_reader
        X_train, y_train = mnist_reader.load_mnist('../research/fashion-mnist/data/fashion', kind='train')
        X_test, y_test = mnist_reader.load_mnist('../research/fashion-mnist/data/fashion', kind='t10k')

    elif (name == "mnist-70k"):
        mnist = pd.read_csv('../research/mnist_784.csv', dtype=np.uint8)

        X_train = np.ascontiguousarray(mnist.iloc[:, :-1].values)
        y_train = np.ascontiguousarray(mnist.iloc[:, -1].values)
        X_test, y_test =  None, None

    elif (name == "mnist-2k"):
        from sklearn import datasets
        digits = datasets.load_digits()
        X_train = digits['data']
        y_train = digits['target']
        X_test, y_test =  None, None

    elif (name == "macosko"):
        pass

    else:
        print('unknown dataset - exiting')
        sys.exit(0)

    return (X_train, y_train, X_test, y_test)


def parameters_difference(parameters):
    res = [(k, v) for k, v in parameters.items() if default_parameters[k] != v]
    return res

