import numpy as np

import scipy.spatial as ss
EPSILON = np.finfo(np.float64).eps



import pynndescent


class NNDescent:

    def build(self, data, k):
        # self.check_metric(self.metric)

        # These values were taken from UMAP, which we assume to be sensible defaults
        n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20))
        n_iters = max(5, int(round(np.log2(data.shape[0]))))

        # UMAP uses the "alternative" algorithm, but that sometimes causes
        # memory corruption, so use the standard one, which seems to work fine
        self.index = pynndescent.NNDescent(
            data,
            n_neighbors=15,
            metric=self.metric,
            # metric_kwds=self.metric_params,
            # random_state=self.random_state,
            n_trees=n_trees,
            n_iters=n_iters,
            algorithm="standard",
            max_candidates=60,
            # n_jobs=self.n_jobs,
        )

        # indices, distances = self.index.query(data, k=k + 1, queue_size=1)
        # return indices[:, 1:], distances[:, 1:]

    def query(self, query, k):
        return self.index.query(query, k=k, queue_size=1)




class Distances:

    def __new__(
        self,
        data,
        method='full',
        metric='euclidean',
        # metric_params=None,
        # symmetrize=True,
        # n_jobs=1,
        # random_state=None,
    ):
        # compute pairwise distances
        if (method == 'full'):
            pdist = ss.distance.pdist(data, metric)
            distances = ss.distance.squareform(pdist)

        elif method == 'approx':
            # use ANN from pynndescent
            raise NotImplementedError("Method not yet supported. Choice are approx or full.")
        else:
            raise ValueError("Method not supported. Choice are approx or full.")

        return distances


class ForceEmbedding(np.ndarray):

    def __new__(cls,
        embedding,
        pdist,
        # random_state=None,
        # optimizer=None,
        # negative_gradient_method="fft",
        # **gradient_descent_params,
    ):
        obj = np.asarray(embedding, dtype=np.float64, order="C").view(ForceEmbedding)

        # add variables
        obj.hd_ij = pdist
        obj.inv_hd_ij_2 = 1./(pdist**2 + EPSILON)

        return obj

    def stress(self):
        pdist = ss.distance.pdist(self)
        ld_ij_2 = ss.distance.squareform(pdist)
        res = np.sum((ld_ij_2 - self.hd_ij)**2 * self.inv_hd_ij_2)
        return res

    def force(self):
        # hd_ij, ld_ij: pairwise distances in high, low dimensional space resp.

        # F_ij = ||Xi-Xj||-hd_ij)/||Xi-Xj|| * (Xi-Xj)/hd_ij**2
        pdist = ss.distance.pdist(self, 'euclidean')
        ld_ij_2 = ss.distance.squareform(pdist)

        # https://stackoverflow.com/questions/32415061/
        a = self[:, None, :]
        ld_ij_1 = (a - a.swapaxes(0,1))

        F_ij_2 = (ld_ij_2 - self.hd_ij) / (ld_ij_2 + EPSILON)
        F_ij = F_ij_2[:,:] * self.inv_hd_ij_2
        # https://stackoverflow.com/questions/39026173/
        F_ij = F_ij[:,:,None] * ld_ij_1

        F_i = np.sum(F_ij, axis=1)

        return F_i

    def optimize(self,
        n_iter,
        learning_rate=200,
        momentum=0.5, exaggeration=None, dof=1, min_gain=0.01,
        min_grad_norm=1e-8, max_grad_norm=None, theta=0.5,
        n_interpolation_points=3, min_num_intervals=50, ints_in_interval=1,
        reference_embedding=None, n_jobs=1,
        use_callbacks=False, callbacks=None, callbacks_every_iters=50
        ):

        update = np.zeros_like(self)
        gains = np.ones_like(self)

        self.error = []
        self.error.append(self.stress())

        for iteration in range(n_iter):

            gradient = self.force()

            grad_direction_flipped = np.sign(update) != np.sign(gradient)
            grad_direction_same = np.invert(grad_direction_flipped)
            gains[grad_direction_flipped] += 0.2
            gains[grad_direction_same] = gains[grad_direction_same] * 0.8 + min_gain

            update = momentum * update - learning_rate * gains * gradient

            self += update

            self.error.append(self.stress())

        return self



def stress(X):
    X = self[:, None, :]
    ld_ij_2 = (X - X.swapaxes(0,1))
    # res = np.sum((ld_ij_2 - self.hd_ij)**2 * self.inv_hd_ij_2)
    res = np.sum((ld_ij_2))
    return res
