import numpy as np

import scipy.spatial as ss
EPSILON = np.finfo(np.float64).eps

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

