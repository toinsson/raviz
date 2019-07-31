import numpy as np

import openTSNE
from openTSNE.affinity import Affinities, joint_probabilities_nn

class TSNEEmbedding(openTSNE.TSNEEmbedding):
    """classic TSNE with stochastic sampling."""
    
    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    
    def __new__(
        cls,
        embedding,
        affinities,
        random_state=None,
        optimizer=None,
        negative_gradient_method="fft",
        **gradient_descent_params,
    ):
        # init_checks.num_samples(embedding.shape[0], affinities.P.shape[0])

        obj = np.asarray(embedding, dtype=np.float64, order="C").view(TSNEEmbedding)

        obj.affinities = affinities  # type: Affinities
        obj.gradient_descent_params = gradient_descent_params  # type: dict
        obj.gradient_descent_params["negative_gradient_method"] = negative_gradient_method
        obj.random_state = random_state

        if optimizer is None:
            optimizer = gradient_descent()
        elif not isinstance(optimizer, gradient_descent):
            raise TypeError(
                "`optimizer` must be an instance of `%s`, but got `%s`." % (
                    gradient_descent.__class__.__name__, type(optimizer)
                )
            )
        obj.optimizer = optimizer

        obj.kl_divergence = None

        return obj

    def optimize_(self, n_iter, inplace=False, propagate_exception=False,
                 **gradient_descent_params):

        if inplace:
            embedding = self
        else:
            embedding = TSNEEmbedding(
                np.copy(self),
                self.affinities,
                random_state=self.random_state,
                optimizer=self.optimizer.copy(),
                **self.gradient_descent_params,
            )

        # If optimization parameters were passed to this funciton, prefer those
        # over the defaults specified in the TSNE object
        optim_params = dict(self.gradient_descent_params)
        optim_params.update(gradient_descent_params)
        openTSNE.tsne._handle_nice_params(optim_params)
        optim_params["n_iter"] = n_iter

        try:
            # Run gradient descent with the embedding optimizer so gains are
            # properly updated and kept
            error, embedding = embedding.optimizer(
                embedding=embedding, affinity=self.affinities, **optim_params
            )

        except openTSNE.OptimizationInterrupt as ex:
            log.info("Optimization was interrupted with callback.")
            if propagate_exception:
                raise ex
            error, embedding = ex.error, ex.final_embedding

        embedding.kl_divergence = error

        return embedding


class StochasticNN(Affinities):

    def __init__(self, 
        data,
        perplexity=30,
        method="approx",
        metric="euclidean",
        metric_params=None,
        symmetrize=True,
        n_jobs=1,
        random_state=None,

        Vmax=5, Smax=10
        ):

        self.data = data
        self.N = data.shape[0]
        self.M = data.shape[1]

        self.perplexity = perplexity
        self.n_jobs = n_jobs
        self.symmetrize = symmetrize


        self.Vmax = Vmax
        self.Smax = Smax

        self.neighbors = np.zeros((self.N, Vmax), dtype=np.int32)
        self.distances = np.zeros((self.N, Vmax))


        # init with full random sampling
        # not enough mem ...
        #         self.neighbors = np.random.choice(self.N, size=self.N*Vmax, replace=True).reshape(-1,Vmax).astype(np.int32)
        #         diff = self.data.reshape((self.N,1,self.M))-self.data[self.neighbors]
        #         self.distances = np.linalg.norm(diff, axis=2)
        for i, point in enumerate(data):
            neighbors = np.random.choice(self.N, size=Vmax, replace=True).astype(np.int32)
            self.neighbors[i] = neighbors
            self.distances[i] = np.linalg.norm(point-data[neighbors], axis=1)


        self.P = joint_probabilities_nn(
            self.neighbors,
            self.distances,
            [self.perplexity],
            symmetrize=self.symmetrize,
            n_jobs=self.n_jobs,
        )


    def recompute_P(self):

        # very slow if in loop
        ids = np.random.choice(self.N, size=(self.N, 3*self.Smax), replace=True)

        # for each object
        for i, point in enumerate(self.data):

            # sample randomly N points
            candidates = self.data[ids[i]]

            # compute candidate distance
            cd = np.linalg.norm((point-candidates), axis=1)

            # sort best and udpate
            best_n = np.argsort(np.hstack([self.distances[i], cd]))
            self.neighbors[i] = np.hstack([self.neighbors[i], ids[i]])[best_n[:self.Vmax]]
            self.distances[i] = np.hstack([self.distances[i], cd])[best_n[:self.Vmax]]

        self.P = joint_probabilities_nn(
            self.neighbors,
            self.distances,
            [self.perplexity],
            symmetrize=self.symmetrize,
            n_jobs=self.n_jobs,
        )

        return self.P


from collections import Iterable


class gradient_descent(openTSNE.tsne.gradient_descent):
    def __init__(self):
        self.gains = None

    def copy(self):
        optimizer = self.__class__()
        if self.gains is not None:
            optimizer.gains = np.copy(self.gains)
        return optimizer

    def __call__(self, embedding, affinity, n_iter, objective_function, learning_rate=200,
                 momentum=0.5, exaggeration=None, dof=1, min_gain=0.01,
                 min_grad_norm=1e-8, max_grad_norm=None, theta=0.5,
                 n_interpolation_points=3, min_num_intervals=50, ints_in_interval=1,
                 reference_embedding=None, n_jobs=1,
                 use_callbacks=False, callbacks=None, callbacks_every_iters=50):

        assert isinstance(embedding, np.ndarray), \
            "`embedding` must be an instance of `np.ndarray`. Got `%s` instead" \
            % type(embedding)

        if reference_embedding is not None:
            assert isinstance(reference_embedding, np.ndarray), \
                "`reference_embedding` must be an instance of `np.ndarray`. Got " \
                "`%s` instead" % type(reference_embedding)

        update = np.zeros_like(embedding)
        if self.gains is None:
            self.gains = np.ones_like(embedding)

        bh_params = {"theta": theta}
        fft_params = {"n_interpolation_points": n_interpolation_points,
                      "min_num_intervals": min_num_intervals,
                      "ints_in_interval": ints_in_interval}


        # # Lie about the P values for bigger attraction forces
        # if exaggeration is None:
        #     exaggeration = 1

        # if exaggeration != 1:
        #     P *= exaggeration

        # Notify the callbacks that the optimization is about to start
        if isinstance(callbacks, Iterable):
            for callback in callbacks:
                # Only call function if present on object
                getattr(callback, "optimization_about_to_start", lambda: ...)()

        for iteration in range(n_iter):

            print("iteration:{}".format(iteration))

            should_call_callback = use_callbacks and (iteration + 1) % callbacks_every_iters == 0
            should_eval_error = should_call_callback

            ## recompute P
            P = affinity.recompute_P()
            # Lie about the P values for bigger attraction forces
            # if exaggeration is None: exaggeration = 1
            # if exaggeration != 1: P *= exaggeration

            # # Lie about the P values for bigger attraction forces
            # if exaggeration is None: exaggeration = 1
            # if exaggeration != 1: P *= exaggeration


            error, gradient = objective_function(
                embedding, P, dof=dof, bh_params=bh_params, fft_params=fft_params,
                reference_embedding=reference_embedding, n_jobs=n_jobs,
                should_eval_error=should_eval_error,
            )

            # Clip gradients to avoid points shooting off. This can be an issue
            # when applying transform and points are initialized so that the new
            # points overlap with the reference points, leading to large
            # gradients
            if max_grad_norm is not None:
                norm = np.linalg.norm(gradient, axis=1)
                coeff = max_grad_norm / (norm + 1e-6)
                mask = coeff < 1
                gradient[mask] *= coeff[mask, None]

            # Correct the KL divergence w.r.t. the exaggeration if needed
            if should_eval_error and exaggeration != 1:
                error = error / exaggeration - np.log(exaggeration)

            if should_call_callback:
                # Continue only if all the callbacks say so
                should_stop = any((bool(c(iteration + 1, error, embedding)) for c in callbacks))
                if should_stop:
                    # Make sure to un-exaggerate P so it's not corrupted in future runs
                    if exaggeration != 1:
                        P /= exaggeration
                    raise openTSNE.OptimizationInterrupt(error=error, final_embedding=embedding)

            # Update the embedding using the gradient
            grad_direction_flipped = np.sign(update) != np.sign(gradient)
            grad_direction_same = np.invert(grad_direction_flipped)
            self.gains[grad_direction_flipped] += 0.2
            self.gains[grad_direction_same] = self.gains[grad_direction_same] * 0.8 + min_gain
            update = momentum * update - learning_rate * self.gains * gradient
            embedding += update

            # Zero-mean the embedding only if we're not adding new data points,
            # otherwise this will reset point positions
            if reference_embedding is None:
                embedding -= np.mean(embedding, axis=0)

            if np.linalg.norm(gradient) < min_grad_norm:
                log.info("Gradient norm eps reached. Finished.")
                break

        # Make sure to un-exaggerate P so it's not corrupted in future runs
        if exaggeration != 1:
            P /= exaggeration

        # The error from the loop is the one for the previous, non-updated
        # embedding. We need to return the error for the actual final embedding, so
        # compute that at the end before returning
        error, _ = objective_function(
            embedding, P, dof=dof, bh_params=bh_params, fft_params=fft_params,
            reference_embedding=reference_embedding, n_jobs=n_jobs,
            should_eval_error=True,
        )

        return error, embedding


