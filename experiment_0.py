import matplotlib.pyplot as plt

import numpy as np

import openTSNE
from openTSNE import affinity

def optimize_embedding_with_init(init, aff, res):
    emb = openTSNE.TSNEEmbedding(init,
                                aff,
                                n_jobs=8,
                                callbacks=lambda it, err, emb: res.append((it, err)),
                                callbacks_every_iters=1,
                                )
    emb_1 = emb.optimize(n_iter=250, exaggeration=12, momentum=0.5)
    emb_2 = emb_1.optimize(n_iter=750, exaggeration=1, momentum=0.8)

    return emb_2


def plot_inits(init_random, init_pca, init_spectral, y_train):

    fig, ax = plt.subplots(1,3, figsize=(18,4))

    tmp = zip([init_random, init_pca, init_spectral], ['random', 'pca', 'spectral'])

    for i, (data, title) in enumerate(tmp):
        im = ax[i].scatter(data[:,0], data[:,1], c=y_train, rasterized=True)
        ax[i].set_title(title)
        _=fig.colorbar(im, ax=ax[i])


def plot_inits_emb(tsne_init_random, tsne_init_pca, tsne_init_spectral, y_train):
    fig, ax = plt.subplots(1,3, figsize=(18,4))

    tmp = zip([tsne_init_random, tsne_init_pca, tsne_init_spectral], ['random', 'pca', 'spectral'])

    for i, (data, title) in enumerate(tmp):
        im = ax[i].scatter(data[:,0], data[:,1], c=y_train, rasterized=True)
        ax[i].set_title(title)
        _=fig.colorbar(im, ax=ax[i])


def plot_inits_KL(tsne_inits_res_np):
    fig,ax = plt.subplots(1,3, figsize=(18,4))
    ax[0].plot(tsne_inits_res_np[:1000,1])
    ax[0].plot(tsne_inits_res_np[1000:2000,1])
    ax[0].plot(tsne_inits_res_np[2000:,1])
    ax[0].legend(['random', 'pca', 'spectral'])
    ax[0].set_title('KL divergence')

    ax[1].plot(tsne_inits_res_np[:250,1])
    ax[1].plot(tsne_inits_res_np[1000:1250,1])
    ax[1].plot(tsne_inits_res_np[2000:2250,1])
    ax[1].legend(['random', 'pca', 'spectral'])
    ax[1].set_title('KL divergence for first 250 steps\n EE=12, momentum=0.5')

    x = np.arange(250, 1000)
    ax[2].plot(x, tsne_inits_res_np[250:1000,1])
    ax[2].plot(x, tsne_inits_res_np[1250:2000,1])
    ax[2].plot(x, tsne_inits_res_np[2250:,1])
    ax[2].legend(['random', 'pca', 'spectral'])
    _=ax[2].set_title('KL divergence for last 750 steps\n EE=1, momentum=0.8')


from fa2 import ForceAtlas2

def compare_tsne_force_mnist_2k_70k(X_train_2k, X_train_70k):

    # tsne
    tnse = openTSNE.TSNE(n_jobs=8)
    tsne_2k = tnse.fit(X_train_2k)

    tnse = openTSNE.TSNE(n_jobs=8)
    tsne_70k = tnse.fit(X_train_70k)

    # force atlas
    affinities = affinity.PerplexityBasedNN(X_train_2k)
    init_pca = openTSNE.initialization.pca(X_train_2k)
    forceatlas2 = ForceAtlas2(verbose=False)
    positions = forceatlas2.forceatlas2(affinities.P, pos=init_pca, iterations=1000)
    fa2_2k = np.array(positions)

    affinities = affinity.PerplexityBasedNN(X_train_70k)
    init_pca = openTSNE.initialization.pca(X_train_70k)
    forceatlas2 = ForceAtlas2(verbose=False)
    positions = forceatlas2.forceatlas2(affinities.P, pos=init_pca, iterations=1000)
    fa2_70k = np.array(positions)

    return tsne_2k, tsne_70k, fa2_2k, fa2_70k

















