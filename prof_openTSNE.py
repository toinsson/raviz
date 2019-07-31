
import time

import pandas as pd
import numpy as np


import logging
import daiquiri
daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)
logger.info("It works and log to stderr by default with color!")




# load the dataset
def get_dataset(name):
    """from name return (X_train, y_train, X_test, y_test) if exist otherwise None for missing
    """
    # check the dataset name: mnist, mnist-fashion, macosko
    import sys
    sys.path.insert(0, "../research/fashion-mnist/")
    from utils import mnist_reader
    X_train, y_train = mnist_reader.load_mnist('../research/fashion-mnist/data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('../research/fashion-mnist/data/fashion', kind='t10k')

    return (X_train, y_train, X_test, y_test)


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


parameters = default_parameters
parameters['n_jobs'] = 8
parameters['n_iter'] = 8

   

class ProfilingData(object):
        # data = None
        # params = None
    def __init__(self, params):
        self.params = params

        self.data_ = None
    # def record_data(self):
    # # self.data = data

    @property
    def data(self):
        return self.data_

    @data.setter
    def data(self, value):
        self.data_ = value


class Callback:

    def __init__(self):
        self.data = np.zeros((0,3))

    def optimization_about_to_start(self):
        logger.info("starting")

    def __call__(self, iteration, error, embedding):
        logger.info("cb {} {}".format(iteration, error))
        self.data = np.vstack((self.data, [time.time(), iteration, error]))


from openTSNE import TSNE, TSNEEmbedding, affinity, initialization
from openTSNE import initialization
from openTSNE.callbacks import ErrorLogger


tsne = TSNE(
    # perplexity=30,
    # initialization="random",
    # metric="euclidean",
    # n_jobs=8,
    # random_state=42,
    # n_iter=10,
    **parameters
    # The embedding will be appended to the list we defined above, make sure we copy the
    # embedding, otherwise the same object reference will be stored for every iteration
    # callbacks=lambda it, err, emb: print({'it':it, 'err':err}),
    # # This should be done on every iteration
    # callbacks_every_iters=1,
)

# e.create("SESSION", "RunA", data=tsne.__dict__)

cb = Callback()

# lambda it, err, emb: opt_res_pca.append((it, err))
# lambda it, err, emb: tmp_rec.append((it, err, time.time())

tsne.callbacks = cb 
tsne.callbacks_every_iters = 1


(X_train, y_train, X_test, y_test) = get_dataset(None)

profdata = ProfilingData(parameters)
# profdata.record_start()


# tsne.fit(X_train)

profdata.data = cb.data.copy()


# save data to file
import dill
with open('data.pkl', 'wb') as output: 
    dill.dump(profdata, output) 


