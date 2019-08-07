import matplotlib.pyplot as plt

import numpy as np

import openTSNE

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