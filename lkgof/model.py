"""Module containing latent variable models"""

import autograd.numpy as np
from abc import ABCMeta, abstractmethod
from lkgof import density, util
from lkgof.density import discrete_score, continuous_score
from lkgof import data
from lkgof import mcmc
from lkgof.data import Data
from lkgof import numpyro_models as nprm
from scipy import stats
import math
from lkgof.util import random_choice_prob_index


class LatentVariableModel(object, metaclass=ABCMeta):
    """
    An abstract class of latent variable models.
    """
    
    @property
    @abstractmethod
    def var_type_disc(self):
        """Indicates if the observed variable is discrete
        or not. If discrete, returns True.
        """
        pass

    def get_unnormalized_density(self):
        """
        Return a `lkgof.density.UnnormalizedDensity`
        if the model has it.
        Return None if not available.
        """
        pass

    @property
    @abstractmethod
    def dim(self):
        """
        Return the dimension of the input.
        """
        raise NotImplementedError()

    @abstractmethod
    def sample(self, n, seed=3):
        """Returns samples"""
        raise NotImplementedError()

    @abstractmethod
    def log_unnormalizedjoint(self, X, latents):
        """Returns the unnoramlized joint density
        evaluated at X and latents
        """
        pass

    def score_joint(self, X, latents): 
        """Evaluate score of the joint at X and latents. 
        The score is averaged over latent samples provided there 
        are multiple of them.

        Subclasses may extend this if that is more effcient.

        This method computes the derivative of
        p(x, latents) w.r.t. X 

        Args:
            X (numpy.ndarray): array of size n x d 
            latents (dict): dictionary containing latents and parameters

        Returns:
            numpy array of size (n, d)
        """
        log_joint = self.log_unnormalizedjoint
        if self.var_type_disc:
            n_values = self.n_values
            return discrete_score(log_joint,
                                  X, n_values, latents,
                                  average=True,
                                  )     
        else:
            return continuous_score(log_joint, X, latents,
                                    average=True)

    def get_datasource(self):
        """
        Return a `kgof.data.DataSource` using sample method 
        and lkgof.DynamicDataSource.
        A DataSource allows one to sample from the model.
        """
        ds = data.DynamicDataSource(self.sample)
        return ds

    def has_datasource(self):
        """
        Return true if this model can provide a `kgof.data.DataSource` which
        allows sample generation.
        """
        return self.get_datasource() is not None

    def has_posterior(self):
        return self.get_posterior() is not None
    
    def has_unnormalized_density(self):
        return self.get_unnormalized_density() is not None
    
    def posterior(self, X, n_sample, 
                  seed=13,
                  *args, **kwargs):
        """Sampling from the model's 
        posterior distribution. 
        
        Optional method. If the model has 
        the posterior distribution, subclasses 
        can implement a sampling function. 

        Args:
            X (numpy.ndarray): 
                array of size n x d, observation
            n_sample (int): the sample size
            seed (int, optional): Seed. Defaults to 13.
        
        Returns:
            dict of arrays of size n_sample x n x dz,
            where dz is the dimensionality of 
            a latent variable
        """
        pass


class PPCA(LatentVariableModel):
    """LatentVariableModel class implementing PPCA
        Args:
            weight: a weight matrix for the likelihood
            var: the variance parameter for the likelihood
            dim: the dimentionality of the observable variable
            dim_l: the dimensionality of the latent variable
    """

    var_type_disc = False

    def __init__(self, weight, var):
        self.weight = weight
        self.var = var
        self.dim_ = weight.shape[0]
        self.dim_l = weight.shape[1]

    def get_unnormalized_density(self):
        mean = np.zeros(self.dim)
        weight = self.weight
        cov = self.var*np.eye(self.dim) + np.dot(weight, weight.T)
        return density.Normal(mean, cov)

    def sample(self, n, seed=3):
        p = self.get_unnormalized_density()
        ds = p.get_datasource()
        return ds.sample(n, seed=seed)

    def get_numpyro_model(self):
        return nprm.PPCA(self.weight, self.var)
    
    def log_unnormalizedjoint(self, X, latents):
        Z = latents['latent']
        mean = np.dot(Z, self.weight.T)
        log_joint = -0.5*np.sum((X-mean)**2, axis=-1)/self.var
        log_joint = log_joint - 0.5*np.sum(Z**2, axis=-1)
        return log_joint

    @property 
    def dim(self):
        return self.dim_
    
    def posterior(self, X, n_sample, seed=13,
                  n_burnin=200):
        np_model = self.get_numpyro_model()
        return np_model.infer_latent(X, num_samples=n_sample, 
                                     num_warmup=n_burnin,
                                     seed=seed)


class BetaBinomSinglePrior(LatentVariableModel):
    """
    (Multivariate) Beta Binomial Model with a single 
    shared success probability parameter 

    """
    var_type_disc = True

    def __init__(self, n_trials, alpha, beta):
        self.n_trials = n_trials
        self.n_values = n_trials + 1
        self.dim_ = len(n_trials)
        self.alpha = alpha
        self.beta = beta
        self.ds_pz = data.DSIndMultivariateBeta([alpha], [beta])

    def get_unnormalized_density(self):
        return density.BetaBinomSinglePriorMarginal(
            self.n_trials, self.alpha, self.beta)

    @property
    def dim(self):
        return self.dim_
    
    def log_unnormalizedjoint(self, X, latents):
        probs = latents['success_probs'].squeeze()
        cond_den = stats.binom(self.n_trials, probs).logpmf(X)
        log_joint = np.sum(np.log(probs)) + np.sum(cond_den, axis=-1)
        return log_joint

    def sample(self, n, seed=3):
        d = self.dim
        pvals = self.ds_pz.sample(d, seed).data()
        assert len(self.n_trials) == len(pvals)
        with util.NumpySeedContext(seed):
            X = np.random.binomial(self.n_trials,
                                   pvals, [n, d])
        return Data(X)

    def posterior(self, X, n_sample, seed=15,
                  *args, **kwargs):
        n, d = X.shape
        alphas = X + self.alpha
        betas = (self.n_trials-X) + self.beta
        with util.NumpySeedContext(seed):
            probs = stats.beta.rvs(alphas, betas,
                                   size=[n_sample, n, d])
        return {'success_probs': probs}


class LDAEmBayes(LatentVariableModel):
    """LDA Model.

    Args:
        alpha (np.ndarray):
            Sparsity Parameters: Array of size K (K = number of topics)
        beta (np.ndarray):
            Array of K x V (V = vocab size) representing a collection of
            distributins over words
        n_values:
            Array of shape (W,) (W = number of words in a document)
    """
    
    def __init__(self, alpha, beta, n_values):
        self.alpha = alpha
        self.beta = beta
        self.n_topics = len(alpha)
        self.n_values = n_values
        self.vocab_size = int(n_values[0])
        assert beta.shape[0] == len(alpha)
        assert beta.shape[1] == self.vocab_size
        self.dim_ = len(n_values)
    
    def var_type_disc(self):
        return
    
    @property
    def dim(self):
        return self.dim_

    def log_unnormalizedjoint(self, X, latents):
        """log of the likelihood p(x | z)

        Note that this is not log of joint.
        We are ignoring the Dirichlet prior
        term, which is not needed to compute
        the score function.

        Args:
            X (numpy.ndarray): array of n x d
            latents (dict): dictionary of latents

        Returns:
            numpy.ndarray: array of size [n, ]
        """ 
        # n words for each document X[i]
        n_docs, n_words = X.shape
        vocab_size = self.vocab_size
        # n_topics x vocab_size
        beta = self.beta
        Z = latents['Z']
        n_samples = Z.shape[0]

        word_probs = np.empty([n_samples, n_docs*n_words])
        X_ = X.reshape(n_docs*n_words)
        range_dn = np.arange(n_docs*n_words)
        for i in range(n_samples):
            word_probs[i] = (
                beta[Z[i]].reshape([n_docs*n_words, vocab_size])[range_dn, X_]
            )

        # The implmentation below could be memory expensive
        # n_sample x n_docs x n_words x vocab_size
        # probs = beta[Z]
        # word_probs = probs.reshape(-1, n_docs*n_words, vocab_size)[:, range(n_docs*n_words), X.reshape(n_docs*n_words)]
        return np.sum(np.log(word_probs).reshape(-1, n_docs, n_words), axis=2)


    def score_joint(self, X, latents): 
        """Evaluate score of the joint at X and latents. 
        The score is averaged over latent samples provided there 
        are multiple samples.

        This method computes the score function of x
        with p(x, latents) 

        Args:
            X (numpy.ndarray): array of size n x d 
            latents (dict): dictionary containing latents and parameters

        Returns:
            numpy array of size (n, d)
        """
        n_values = self.n_values.reshape([1, -1])
        beta = self.beta 
        Z = latents['Z']
        n_samples = Z.shape[0] if len(Z.shape)==3 else 1
            
        n_docs, n_words = X.shape

        X_ = np.mod(X+1, n_values).astype(np.int64)
        # the following is possible because of conditional independence
        S = np.zeros([n_docs, n_words])
        for i in range(n_samples):
            log_word_prob = np.log(beta[Z[i], X])
            shift_log_word_prob = np.log(beta[Z[i], X_])
            score_sample = (
                np.exp(shift_log_word_prob - log_word_prob) - 1.
            )
            S = S + (score_sample-S)/(i+1)
        return S

    def posterior(self, X, n_sample, seed=13,
                  n_burnin=5000):
        alpha = self.alpha
        beta = self.beta
        n_topics = self.n_topics
        n_docs, n_words = X.shape
        lda_collapsed_gibbs = mcmc.lda_collapsed_gibbs
        
        with util.NumpySeedContext(seed+18):
            _, Z = self.sample(n_docs, return_latent=True)
            Z_init = Z.data()
            # Z_init = np.random.randint(0, n_topics, [n_docs, n_words])
        Z_batch = lda_collapsed_gibbs(X, n_sample, Z_init,
                                      n_topics, alpha, beta, seed,
                                      n_burnin=n_burnin)
        return {'Z': Z_batch}
                            
    def sample(self, n, seed=13, return_latent=False):
        alpha = self.alpha
        beta = self.beta
        n_topics = len(alpha)
        n_words = self.dim

        drclt = stats.dirichlet
        if return_latent:
            Z_ = []
        with util.NumpySeedContext(seed):
            theta = drclt(alpha).rvs(size=n)
            X = np.empty([n, n_words], dtype=np.int64)
            for i in range(n):
                Z = np.random.choice(n_topics, size=n_words, p=theta[i])
                if return_latent:
                    Z_.append(Z)
                phi_ = beta[Z]
                X[i] = random_choice_prob_index(phi_)
        if return_latent:
            return Data(X), Data(np.array(Z_))
        return Data(X)


class DPMIsoGaussBase(LatentVariableModel):
    """Gaussian Dirichlet Process Mixture model.

    Args:
        var (float):
            Variance of Gaussian likelihood
        prior_mean (np.ndarray):
            Gaussian Prior mean
        prior_var (float):
            Gaussian prior variance
        obs (np.ndarray):
            Observed data of size n_obs x d. Defaults to None.
    """

    var_type_disc = False

    def __init__(self, var, prior_mean, prior_var, obs=None):
        super(DPMIsoGaussBase, self).__init__()
        self.prior_mean = prior_mean
        self.dim_ = len(prior_mean)
        self.prior_var = prior_var
        self.var = var
        self.obs = obs

    def dim(self):
        return self.dim_

    @property
    def is_conditioned(self):
        return self.obs is not None

    def sample(self, n, seed=3, n_burnin=2000):
        if not self.is_conditioned:
            mean = self.prior_mean
            var = self.var + self.prior_var
            ds = data.DSIsotropicNormal(mean, var)
            return ds.sample(n, seed)
        
        n_obs = self.obs.shape[0]
        d = self.dim()
        with util.NumpySeedContext(seed):
            loc_init = (np.random.random((n_obs, d))-0.5)
        locs = mcmc.dpm_isogauss_gibbs(self.obs, n, loc_init, self,
                                       seed=13, n_burnin=n_burnin,
                                       skip=50)
        X = np.empty([n, d])
        with util.NumpySeedContext(seed):
            idx = np.random.randint(0, n_obs+1, size=[n])
            prior_idx = (idx==0)
            n_ = int(np.sum(prior_idx))
            mean = ((self.prior_var**0.5)*np.random.randn(n_, d)
                    + self.prior_mean)
            X[prior_idx] = ((self.var**0.5)*np.random.randn(n_, d)
                            + mean)
            latent_idx = (idx!=0)
            idx_ = idx[latent_idx] - 1
            X[latent_idx] = ((self.var**0.5)*np.random.randn(n-n_, d)
                             + locs[latent_idx, idx_])
        return Data(X)

    def log_unnormalizedjoint(self, X, latents):
        locs = latents['locs']
        var = self.var
        return -0.5 * (np.sum((X-locs)**2, axis=-1)/var)
    
    def marginal_den(self, X):
        d = self.dim()
        mean = self.prior_mean
        var = self.var + self.prior_var
        den = np.exp(-0.5*np.sum((X-mean)**2, axis=-1)/var)
        den /= np.sqrt(2.*math.pi*var)**d
        return den

    def likelihood(self, X, locs):
        # assert X.shape[0] == locs.shape[0]
        assert X.shape[-1] == locs.shape[-1]
        if X.shape[0] == locs.shape[0]:
            diff2 = (X-locs)**2
        else:
            diff2 = np.expand_dims(X, 1) - np.expand_dims(locs, 0)
            diff2 = diff2**2

        var = self.var
        d = self.dim()
        likelihood = np.exp(-(np.sum(diff2, axis=-1))/(2.*var))
        likelihood /= np.sqrt(2.*math.pi*var)**d
        return likelihood
    
    def posterior_mean_cov(self, X):
        """Returns the posterior meea and covariance 
        of the base measure condition on a single observation"""
        d = self.dim()
        var = self.var
        pvar = self.prior_var
        pmean = self.prior_mean
        postvar = var * pvar / (var + pvar)
        mean = postvar * (X/var + pmean/pvar)
        return mean, postvar*np.eye(d)

    def posterior(self, X, n_sample, seed=13,
                  n_burnin=300):
        locs = mcmc.dpm_isogauss_score_posterior(
            X, n_sample, self, 
            n_burnin=n_burnin, seed=seed)
        return {'locs': locs}


def main():
    from lkgof import mcmc
    n_doc = 10
    n_words = 1000
    n_topics = 3
    vocab_size = 1000
    alpha = np.ones(n_topics)
    drclt = stats.dirichlet(alpha=np.ones(vocab_size))
    beta = drclt.rvs(size=n_topics)
    lda = LDAEmBayes(alpha, beta, vocab_size*np.ones(n_words))
    X, Z = lda.sample(n_doc, return_latent=True)
    X = X.data()
    Z_init = np.random.randint(0, n_topics, [n_doc, n_words])
    with util.ContextTimer() as t:
        Z_batch = mcmc.lda_collapsed_gibbs(X, 3, Z_init, n_topics,
                                 alpha, beta, n_burnin=400)
    score_new = lda.score_joint(X, latents={'Z':Z_batch})
    score_old = super(LDAEmBayes, lda).score_joint(X, latents={'Z':Z_batch})
    print(np.abs(score_old - score_new).mean())



if __name__ == '__main__':
    main()
