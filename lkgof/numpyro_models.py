import numpyro
import numpy as onp
import jax.numpy as np
import numpyro.distributions as dist
import numpyro.handlers as handlers
from numpyro.infer import MCMC, NUTS
from abc import ABCMeta, abstractmethod
from jax import random
from jax import vmap, jit, grad

from future.utils import with_metaclass


class NumPyroModel(with_metaclass(ABCMeta, object)):
    """
    abstract class of latent variable models implemented in NumPyro
    """

    @abstractmethod
    def score(self, X, num_samples=500, num_warmup=200,
              return_samples=False, seed=13):
        """
        Args:
            - X: a n x d ndarray
            - num_samples: the number of MC sample points
            - num_warmup: warm up size
            - return_samples: Bool.
        Returns
            - approximate score; n x d ndarray
            - if MC sample scores return_samples is True
                ndarray of size num_samples x n x d
        """
        raise NotImplementedError()

    @abstractmethod
    def model(self):
        """
        A callable for model description
        """
        raise NotImplementedError


class PPCA(NumPyroModel):
    """
    Args:
        - weight: a weight matrix for the likelihood
        - var: the variance parameter for the likelihood
        - dim: the dimentionality of the observable variable
        - dim_l: the dimensionality of the latent variable
    """
    def __init__(self, weight, var):
        self.weight = weight
        self.var = var
        self.dim = weight.shape[0]
        self.dim_l = weight.shape[1]

    def model(self, obs, rng_seed=13):
        weight = self.weight
        var = self.var
        dx, dz = weight.shape

        N = obs.shape[0]
        with handlers.seed(rng_seed=rng_seed):
            with numpyro.plate('N', N):
                Z = numpyro.sample(
                    'latent', dist.MultivariateNormal(
                        loc=0, covariance_matrix=np.eye(dz))
                    )
                mean = np.dot(Z, weight.T)
                cov = var * np.eye(dx)
                numpyro.sample(
                    'obs',
                    dist.MultivariateNormal(mean, covariance_matrix=cov),
                    obs=obs
                )

    def score(self, X, num_samples=500, num_warmup=200,
              return_samples=False, seed=13,
              batch_size=20):
        """
        Args:
            - X: a n x d ndarray
            - num_samples: the number of MC sample points
            - num_warmup: warm up size
            - return_samples: Bool
        Returns
            - approximate score; n x d ndarray
            - if MC sample scores return_samples is True
                ndarray of size num_samples x n x d
        """
        @jit
        def iso_gaussian_score(mean, X):
            def log_den(X, mean):
                return np.sum(-(X-mean)**2) / (2.0*self.var)
            return vmap(grad(log_den), in_axes=(0, 0))(X, mean)

        @jit
        def joint_score(batched_mean, X):
            return vmap(iso_gaussian_score, in_axes=(0, None))(batched_mean, X)

        n, d = X.shape
        X_ = np.array(X)
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples,
                    num_warmup=num_warmup, progress_bar=False)
        mcmc.run(random.PRNGKey(seed), X_)
        batched_posterior = mcmc.get_samples()['latent']
        batched_posterior = np.dot(batched_posterior, self.weight.T)

        joint_scores = joint_score(batched_posterior, X_)
        estimated_score = joint_scores.mean(axis=0)
        if not return_samples:
            return onp.array(estimated_score)

        assert num_samples > batch_size
        num_batches = num_samples // batch_size
        # TODO: We might want to impose num_batches >= d:
        n_ = num_batches * batch_size
        # np.array_split has not been implemented
        batched_joint_scores = np.array(
            np.split(joint_scores[:n_], num_batches, axis=0))
        batch_size = batched_joint_scores.shape[1]
        joint_score0 = np.mean(batched_joint_scores, axis=1) - estimated_score
        return onp.array(estimated_score), onp.array(joint_score0)

    def infer_latent(self, X, num_samples=500, num_warmup=200, init_params=None, seed=17):
        n, d = X.shape
        X_ = np.array(X)
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples,
                    num_warmup=num_warmup, progress_bar=False)
        mcmc.run(random.PRNGKey(seed), X_, init_params=init_params)
        Z = mcmc.get_samples()['latent']
        return {'latent': onp.array(Z)}
