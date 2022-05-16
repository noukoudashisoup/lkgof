from kgof.data import *
from lkgof import util
import scipy.stats as stats
import scipy.special as special
import numpy as np


class DSMultivariateBern(DataSource):
    """
    A DataSource implementing a multivariate bernoulli
    distribution.
    """

    def __init__(self, probs):
        """
        probs = (d, ) numpy array
        """

        self.probs = probs
        self.d = len(probs)

    def sample(self, n, seed=3):
        sampler = stats.bernoulli.rvs
        d = self.d
        with util.NumpySeedContext(seed=seed):
            X = sampler(p=self.probs, size=[n, d]).astype(np.int)
        return Data(X)


class DSIndMultivariateBeta(DataSource):
    """
    A stack of d Beta distributions, each can have its own parameters (alpha,
    beta).
    """

    def __init__(self, alphas, betas):
        assert len(alphas) == len(betas)
        self.alphas = alphas
        self.betas = betas
        self.d = len(alphas)

    def sample(self, n, seed=3):
        alphas = self.alphas
        betas = self.betas
        with util.NumpySeedContext(seed):
            X = stats.beta.rvs(alphas, betas, size=n)
        return Data(X)


class DSMultivariateBinomial(DataSource):
    """
    A stack of d Binomial distributions, each can have its own n and p.
    """

    def __init__(self, n_trials, probs):
        """
        Args:
        - n_trials: 1-d array of length d containing values for n's
        - probs: 1-d array of length d containing values for probabilities, one
          for each Binomial
        """
        assert len(n_trials) == len(probs)
        self.n_trials = n_trials
        self.probs = probs
        self.d = len(n_trials)

    def sample(self, n, seed=3):
        n_trials = self.n_trials
        p = self.probs
        d = self.d
        with util.NumpySeedContext(seed):
            X = [stats.binom.rvs(n=n_trials[j], p=p[j], size=n)
                 for j in range(d)]
        X = np.array(X).T
        return Data(X)


class DynamicDataSource(DataSource):
    """
    DataSource object whose sample method is given 
    in the constructor.
    Attribute:
        - sample: sample method. If a bound method 
        is given, this class will use it. 
    """

    def __init__(self, sample):
        setattr(self, 'sample', sample)
        
    def sample(self):
        pass


class DSMixtureDist(DataSource):
    """ DataSource object for mixtures of distributions.
    Args:
    - data_sources: a list of data sources
    - pmix: numpy array of mixture weights
    """

    def __init__(self, data_sources, pmix):
        self.data_sources = data_sources
        self.pmix = pmix

    def sample(self, n, seed=3):
        pmix = self.pmix
        data_sources = self.data_sources
        X = []
        with util.NumpySeedContext(seed=seed):
            idx = np.random.choice(len(data_sources), size=[n], p=pmix)
        for i in range(n):
            ds = data_sources[idx[i]]
            x = ds.sample(1, seed+i)
            X.append(x.data().flatten())
        return Data(np.array(X))


class DSCategorical(DataSource):
    """
    DataSource object for multimonial distributions

    Attributes:
        - n_trials: a integer scalar variable for number of trials
        - probs: success probability for each event k
    """

    def __init__(self, probs):
        self.probs = probs
        if len(probs.shape) > 1:
            self._n_values = probs.shape[1]
        if len(probs.shape) == 1:
            self._n_values = len(probs)
            
    def sample(self, n, seed=3):
        p = self.probs
        with util.NumpySeedContext(seed=seed):
            X = util.choice(p, n)
        return Data(X)


class DSMultinomial(DataSource):
    """
    DataSource object for multimonial distributions

    Attributes:
        - n_trials: a integer scalar variable for number of trials
        - probs: success probability for each event k
    """

    def __init__(self, n_trials, probs):
        self.n_trials = n_trials
        self.probs = probs

        if np.abs(np.sum(self.probs) - 1) > 1e-8:
            raise ValueError('Probabilities do not sum to 1.')

    def sample(self, n, seed=3):
        with util.NumpySeedContext(seed=seed):
            X = stats.multinomial.rvs(size=[n], n=n_trials,
                                      p=self.probs)
        return Data(X)


class DSSigmoidBeliefNet(DataSource):
    """
    DataSource object for multimonial distributions

    Attributes:
    - W: d x d numpy array for weight
    - h: numpy array of size [d,] for hidden state
    - b: numpy array of size [d,] for bias
    """

    def __init__(self, W, h, b=None):
        self.W = W
        self.h = h
        d = W.shape[0]
        self.b = b if b is not None else np.zeros(d)
        assert len(self.h) == W.shape[1]
        assert len(self.b) == W.shape[0]

    def sample(self, n, seed=3):
        W = self.W
        b = self.b
        d = W.shape[0]
        h = self.h

        exponent = (np.dot(W, h)+b)
        probs = special.expit(exponent)
        sampler = stats.bernoulli.rvs
        X = sampler(p=probs, size=[n, d]).astype(np.float)
        return Data(X)
