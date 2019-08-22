import time

import numpy as np

import dill

from openTSNE import TSNE, TSNEEmbedding, affinity, initialization
from openTSNE import initialization
from openTSNE.callbacks import ErrorLogger



import logging
import daiquiri
daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)
logger.info("It works and log to stderr by default with color!")


import tools
parameters = tools.default_parameters
(X_train, y_train, X_test, y_test) = tools.get_dataset("mnist")



for i in range(1):
    logger.info('Run {}'.format(i))

    # parameters['random_state'] = i
    # parameters['initialization'] = 'random'
    # parameters['early_exaggeration'] = 100*i+1
    # print(parameters['early_exaggeration'])

    profdata = tools.ProfilingData('default_knn_'+str(i), parameters)

    ## TSNE
    tsne = TSNE(**parameters)
    cb = tools.CallbackKnn(logger, y_train)
    tsne.callbacks = cb 
    tsne.callbacks_every_iters = 1

    emb = tsne.fit(X_train)

    ## STORE
    profdata.data = cb.data
    with open('data/mnist/'+profdata.name+'.pkl', 'wb') as output: 
        dill.dump(profdata, output) 


import ipdb; ipdb.set_trace()
