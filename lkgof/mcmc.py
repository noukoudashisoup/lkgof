"""Module containing MCMC methods for latent variable models"""
import numpy as np
import scipy.stats as stats
from lkgof import util
from lkgof.util import random_choice_prob_index
from lkgof import model as lvm


def lda_collapsed_gibbs(X, n_sample, Z_init, n_topics,
                        alpha, beta, seed=13,
                        n_burnin=1):
    """Samples the latents of data X using a collapsed Gibbs
    sampler.

    Args:
        X (np.ndarray): n x d array
        n_sample (int): number of samples
        Z_init (np.ndarray):
            Initialisation for the latents.
            Array of size n x d.
        n_topics (int): number of topics
        alpha (np.ndarray): Array of n_topics.
        beta (np.ndarray): Word distributions. Array of n_topics x vocab_size.
        seed (int, optional): Random seed. Defaults to 13.
        n_burnin (int, optional): Burn-in sample size. Defaults to 1.

    Returns:
        np.ndarray: Array of n x d.
    """
    assert X.shape == Z_init.shape
    n_docs, n_words = X.shape

    def sampling(Z0, n_iter, keepsample=False):
        Z_ = Z0.copy()
        if keepsample:
            Z_batch = np.empty([n_iter, n_docs, n_words],
                               dtype=np.int)
        # topic count
        C = np.zeros([n_docs, n_topics])
        for i_d in range(n_docs):
            u, cnt = np.unique(Z_[i_d], return_counts=True)
            C[i_d, u] = cnt

        for i in range(n_iter):
            j = np.random.randint(0, n_words)
            C[np.arange(n_docs), Z_[:, j]] -= 1
            # p = n_docs x n_topics
            p = beta[:, X[:, j]].T * (C+alpha)
            p = p / np.sum(p, axis=1, keepdims=True)
            Z_[:, j] = random_choice_prob_index(p)
            C[np.arange(n_docs), Z_[:, j]] += 1
            if keepsample:
                Z_batch[i] = Z_.copy()
        if keepsample:
            return Z_batch
        return Z_

    with util.NumpySeedContext(seed):
        Z_ = sampling(Z_init, n_burnin)
        Z_batch = sampling(Z_, n_sample, keepsample=True)
    return Z_batch


def _dpm_isogauss_gibbs_step(X, locs,
                             model, seed=13):
    """A single Gibbs sampling step for dpm_isogauss_gibbs.
    Random-scan Gibbs."""
    assert isinstance(model, lvm.DPMIsoGaussBase)
    n = X.shape[0]
    with util.NumpySeedContext(seed):
        i = np.random.randint(0, n)
        Xi = X[i]
        probs = np.empty(n)
        probs[0] = model.marginal_den(Xi)
        idx = np.arange(n)
        idx = np.delete(idx, i)
        probs[1:] = model.likelihood(X[idx], locs[idx])
        probs /= np.sum(probs)
        sample_idx = np.random.choice(n, size=1, p=probs)
        if sample_idx == 0:
            mean, cov = model.posterior_mean_cov(Xi)
            loc = stats.multivariate_normal.rvs(
                mean=mean, cov=cov)
        else:
            loc = locs[idx][sample_idx-1]
    new_locs = locs.copy()
    new_locs[i] = loc
    return new_locs


def dpm_isogauss_gibbs(X, n_sample, loc_init,
                       model, seed=13,
                       n_burnin=1, skip=1):
    """Samples from the posterior of the latents
    of a given data X according to a DPMIsoGaussBase model.

    Args:
        X (np.ndarray): n x d array
        n_sample (int): the number of samples
        loc_init (np.ndarray): Initialisation for the latents.
        model (lvm.DPMIsoGaussBase): DPMIsoGaussBase object
        seed (int, optional): Random seed. Defaults to 13.
        n_burnin (int, optional): Burn-in sample size. Defaults to 1.
        skip (int, optional): Sample every "skip" points. Defaults to 1.

    Returns:
        np.ndarray: Array of latents of size n x d
    """
    locs = loc_init.copy()
    # burn-in
    for i in range(n_burnin):
        locs = _dpm_isogauss_gibbs_step(X, locs,
                                        model,
                                        seed=seed+i)
    # sampling
    samples = np.empty((n_sample,) + locs.shape)
    for i in range(n_sample):
        for j in range(skip):
            locs = _dpm_isogauss_gibbs_step(
                X, locs, model, seed+i+21+n_burnin)
        samples[i] = locs
    return samples


def dpm_isogauss_score_gibbs(X, n_sample, loc_init,
                             model, seed=13, n_burnin=1):
    if not model.is_conditioned:
        raise ValueError('model needs to be conditioned')
    obs = model.obs
    n_obs, d = obs.shape
    n = X.shape[0]
    samples = np.empty((n_sample,)+loc_init.shape)

    # augment the latent of training data
    with util.NumpySeedContext(seed):
        loc_init_obs = np.random.random([n, n_obs, d])-0.5
        # loc_init_obs = np.random.random([n_obs, d])-0.5

    for i in range(n):
        Xi = X[i].reshape([1, -1])
        X_ = np.concatenate([obs, Xi], axis=0)
        loc_init_ = np.vstack([loc_init_obs[i], loc_init[i][np.newaxis]])
        isamples = dpm_isogauss_gibbs(X_, n_sample, loc_init_,
                                      model, seed=seed, n_burnin=n_burnin)
        # discard the latent of training data
        samples[:, i, :] = isamples[:, -1, :]
    return samples


def dpm_isogauss_metropolis_step(Z_current, X, Z_obs, model, seed=13):
    """Metropolis step for sampling from the posterior
    required for the score function

    Args:
        Z_current (np.ndarray): current position array of size n x d
        X (np.ndarray): array of input (test points) size n x d
        Z_obs (Z_obs): sample from the posterior mean measure n_obs x d
        model (model.Late): DPMIsoGaussBase model
        seed (int, optional): Randodm seed. Defaults to 13.

    Returns:
        [np.ndarray]: next latent sample n x d
        [np.ndarray]: next training data latent sample n_obs x d
    """
    assert isinstance(model, lvm.DPMIsoGaussBase)
    n_obs = Z_obs.shape[0]
    n, d = X.shape
    mean = model.prior_mean
    var = model.prior_var

    # sample from proposal (uniform w.r.t. the poster mean measure)
    Z_proposal = np.empty([n, d])
    Z_obs_ = Z_obs.copy()
    with util.NumpySeedContext(seed+1408):
        for i in range(n):
            Z_obs_ = _dpm_isogauss_gibbs_step(model.obs, Z_obs_,
                                              model, seed+i)
            idx = np.random.randint(0, n_obs+1)
            if idx == 0:
                z = var**0.5 * np.random.randn(d) + mean
            else:
                z = Z_obs_[idx-1]
            Z_proposal[i] = z

    # Metropolis step
    f = model.likelihood
    metropo_ratio = f(X, Z_proposal) / f(X, Z_current)
    accept_prob = np.minimum(metropo_ratio, np.ones(n))
    with util.NumpySeedContext(seed+140):
        u = np.random.random(size=[n])
    update_idx = (u < accept_prob)
    Z_new = Z_current.copy()
    Z_new[update_idx] = Z_proposal[update_idx]
    return Z_new, Z_obs_


def dpm_isogauss_score_posterior(X, n_sample, model, Z_init=None,
                                 loc_init=None, seed=13, n_burnin=1):
    """Samples from the posterior for computing the score
    function.

    Args:
        X (np.ndarray): Evaluation data n x d array
        n_sample (int): The number of generated samples
        model (lvm.DPMIsoGaussBase): DPMIsoGaussBase object
        Z_init (np.ndarray, optional):
            Initialisation for the latents of the evaluation data.i
            n x d. Defaults to None.
        loc_init (np.ndarray, optional):
            Initialisation for the latents of the observed data.
            Size = n_obs x d. Defaults to None.
        seed (int, optional): Random seed. Defaults to 13.
        n_burnin (int, optional): Burn-in sample size. Defaults to 1.

    Raises:
        ValueError: If the model is not conditioned on data.

    Returns:
        [np.ndarray]: Generated samples. Size = (n_sample, n, d).
    """
    assert isinstance(model, lvm.DPMIsoGaussBase)
    if not model.is_conditioned:
        raise ValueError('model needs to be conditioned')
    obs = model.obs
    n_obs, d = obs.shape
    n = X.shape[0]
    samples = np.empty((n_sample, n, d))

    # initilisation
    if loc_init is None:
        with util.NumpySeedContext(seed):
            loc_init = (np.random.randn(n_obs, d))

    Z_obs = dpm_isogauss_gibbs(obs, 1, loc_init, model,
                               seed=13+seed, n_burnin=n_burnin,
                               skip=1)
    Z_obs = Z_obs[-1]

    if Z_init is not None:
        Z_current = Z_init.copy()
    else:
        # posterior w.r.t. the base measure
        # (i.e. when there is no observation)
        pmean, pcov = model.posterior_mean_cov(X)
        Z_current = np.dot(np.random.randn(n, d), pcov) + pmean
    # end initialisation

    for j in range(n_burnin):
        Z_curent, Z_obs = dpm_isogauss_metropolis_step(
            Z_current, X, Z_obs, model, seed=seed+j)

    for j in range(n_sample):
        Z_curent, Z_obs = dpm_isogauss_metropolis_step(
            Z_current, X, Z_obs, model, seed=seed+j+n*n_burnin)
        samples[j] = Z_current
    return samples
