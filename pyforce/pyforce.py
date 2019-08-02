# from numba import njit, prange

# import autograd.numpy as np
import numpy as np

import scipy.spatial as ss
EPSILON = np.finfo(np.float64).eps



import pynndescent


class NNDescent:

    n_jobs = 8

    def build(self, data):
        # self.check_metric(self.metric)

        # These values were taken from UMAP, which we assume to be sensible defaults
        n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20))
        n_iters = max(5, int(round(np.log2(data.shape[0]))))

        # UMAP uses the "alternative" algorithm, but that sometimes causes
        # memory corruption, so use the standard one, which seems to work fine
        self.index = pynndescent.NNDescent(
            data,
            n_neighbors=15,
            metric="euclidean",
            # metric_kwds=self.metric_params,
            # random_state=self.random_state,
            n_trees=n_trees,
            n_iters=n_iters,
            algorithm="standard",
            max_candidates=60,
            n_jobs=self.n_jobs,
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

        obj.error = []

        return obj


    def stress(self):
        pdist = ss.distance.pdist(self)
        ld_ij_2 = ss.distance.squareform(pdist)
        res = np.sum((ld_ij_2 - self.hd_ij)**2 * self.inv_hd_ij_2)
        return res

    def force(self):
        # https://math.stackexchange.com/questions/84331/
        # hd_ij, ld_ij: pairwise distances in high, low dimensional space resp.
        # F_ij = ||Xi-Xj||-hd_ij)/||Xi-Xj|| * (Xi-Xj)/hd_ij**2

        pdist = ss.distance.pdist(self, 'euclidean')
        ld_ij_2 = ss.distance.squareform(pdist)
        F_ij_2 = (ld_ij_2 - self.hd_ij) / (ld_ij_2 + EPSILON)
        F_ij = F_ij_2[:,:] * self.inv_hd_ij_2
        # https://stackoverflow.com/questions/32415061/
        a = self[:, None, :]
        ld_ij_1 = (a - a.swapaxes(0,1))
        # https://stackoverflow.com/questions/39026173/
        F_ij = F_ij[:,:,None] * ld_ij_1

        F_i = np.sum(F_ij, axis=1)

        return 4*F_i

    def optimize(self,
        n_iter=100,
        method="gd",
        ):

        if (method == 'gd'):
            optimizer = gd

        elif method == 'sgd':
            optimizer = sgd

        else:
            raise ValueError("Method not supported. Choice are gradient_descent or sgd.")

        cb = lambda x,i,g: self.error.append(self.stress())
        cb(None, None, None)
        optimizer(self.force, self, callback=cb, num_iters=n_iter)

        return self


def gd(grad, x, callback=None, num_iters=200):
    for i in range(num_iters):

        g = grad()
        if callback: callback(x, i, g)

        x -= g/2  # if full gradient, optimisation is unstable
    return x


# https://github.com/HIPS/autograd/blob/6e5ec96993bd7136c8d96cc6bfb11dccd04490aa/autograd/misc/optimizers.py#L33
def sgd(grad, x, callback=None, num_iters=200, step_size=0.1, mass=0.9):
    """Stochastic gradient descent with momentum."""
    velocity = np.zeros(x.shape)
    for i in range(num_iters):
        g = grad()
        if callback: callback(x, i, g)
        velocity = mass * velocity - (1.0 - mass) * g
        # modify x in place: https://stackoverflow.com/questions/12905338/
        x += step_size * velocity

    return x


from sklearn import preprocessing

class ForceEmbeddingKEdge(np.ndarray):

    def __new__(cls,
        embedding,
        ann,
        # random_state=None,
        # optimizer=None,
        # negative_gradient_method="fft",
        # **gradient_descent_params,
    ):
        obj = np.asarray(embedding, dtype=np.float64, order="C").view(ForceEmbeddingKEdge)

        # add variables
        obj.ann = ann

        return obj

    def set_n_d_K(self, n, d, K):
        self.n = n
        self.d = d[:, 1:]
        self.K = K

        self.d_norm = preprocessing.scale(self.d, with_mean=False)


    def find_K(self, Y, K):
        self.K = K
        self.n, self.d = self.ann.query(Y, K + 1)

        self.d = self.d[:, 1:]

        self.d_norm = preprocessing.scale(self.d, with_mean=False)


    def compute_grad_edge_i(self, seed):
        ld = self[self.n[seed]]

        X_ij = ld[0] - ld[1:] + EPSILON
        ld_ij = np.linalg.norm(X_ij, axis=1) + EPSILON

        d_ij = self.d[seed]
        inv_d_ij_2 = 1./(d_ij**2 + EPSILON)

        g = 2 * ((ld_ij - d_ij) / (ld_ij+EPSILON) * inv_d_ij_2)[:, None] * X_ij

        return g

    def update_vertices(self, seed, grad):
        ns = self.n[seed]
        self[ns[0]]  += np.sum(grad, axis=0)
        self[ns[1:]] -= grad


    # @njit(parallel=True)
    def optimise(self, n_iter=100, n_samples=1000):
        # updates_ids = np.random.choice(X_train.shape[0], size=100000, replace=True)

        for i in prange(n_iter):
            updates_ids = np.random.choice(self.shape[0], size=n_samples, replace=True)
            for vertice in updates_ids:
                grad = self.compute_grad_edge_i(vertice)
                self.update_vertices(vertice, grad)






from autograd import grad

class ForceEmbeddingAutoGrad:

    def __init__(self, embedding, pdist):

        self.X = np.asarray(embedding, dtype=np.float64, order="C").view(ForceEmbedding)
        self.hd_ij = pdist
        self.inv_hd_ij_2 = 1./(pdist**2 + EPSILON)
        self.error = []


    # def stress(self, X):
    #     # 4 times slower than scipy version
    #     X = X[:, None, :]
    #     ld_ij_2 = np.linalg.norm((X - X.swapaxes(0,1)), axis=2)
    #     res = np.sum((ld_ij_2 - self.hd_ij)**2 * self.inv_hd_ij_2)
    #     return res

    def optimize(self, n_iter=100, method="gd"):

        if (method == 'gd'):
            optimizer = gd

        elif method == 'sgd':
            optimizer = sgd

        else:
            raise ValueError("Method not supported. Choice are gradient_descent or sgd.")

        def stress(X):
            # 4 times slower than scipy version
            X = X[:, None, :]
            ld_ij_2 = np.linalg.norm((X - X.swapaxes(0,1)), axis=2)
            res = np.sum((ld_ij_2 - self.hd_ij)**2 * self.inv_hd_ij_2)
            return res

        gradient = grad(stress)

        cb = lambda: self.error.append(stress(self.X))
        cb()

        for i in range(n_iter):

            g = gradient(self.X)
            if callback: callback()

            self.X -= g
        # optimizer(gradient, self, callback=cb, num_iters=n_iter)

        return self.X


