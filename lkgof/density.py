import autograd
import autograd.numpy as np
import scipy.special as special
from kgof import density as kgofden
from kgof.density import *
from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass
from scipy import stats
from lkgof import data


"""
Module containing implementaions of density/probabilty mass functions.
"""


def unit_interval_check(probs):
    """raise ValueError if an element of probs is NOT between 0 and 1"""
    unit_interval = np.logical_and(probs <= 1., probs > 0)
    if not np.all(unit_interval):
        idx = np.where(np.logical_not(unit_interval))
        raise ValueError(
            ('probs shouls be between 1 and 0'
             'd = {}, values={}'.format(list(idx), probs[idx])
             )
        )


class DiscreteFunction(with_metaclass(ABCMeta, object)):
    """ Interface for functions with 
    finite domains"""

    @property
    def n_values(self):
        """Gives an array of integers 
        indicating each dimensions's cardinality"""
        
        try:
            if np.isscalar(self._n_values):
                d = self.dim()
                self._n_values = np.array([self._n_values] * d)
            assert len(self._n_values) == self.dim()
            return self._n_values
        except AttributeError:
            raise NotImplementedError(('The number of possible values should'
                                       'be implemented'))

    @abstractmethod
    def dim(self):
        """Return the dimension of the input."""
        raise NotImplementedError()


def discrete_score(log_den, X, n_values, *args, shift=1,
                   average=False):
    """Returns the score of the density of a
    discrete-valued distribution evaluated at X. 
    Note that this is the negative of the score function
    defined in Yang et al., 2018.


    Args:
        log_den (Callable): log of density 
             
        X (np.ndarray): n x d np.ndarray
        n_values (np.ndarray)): integer array of size d
        shift (int, optional):
            Step size to take forward difference. Defaults to 1.
        average: 
            If the score function is evaluated for multiple 
            samples (latent variable), average over them.
            Defaults to False.

    Returns:
        [numpy.ndarray]:
            Score function evaluation. Array of size n x d.
    """
    n, d = X.shape
    S = np.empty([n, d])
    logden_X = log_den(X, *args)
    if average:
        logden_X = np.mean(logden_X, axis=0)
    for j in range(d):
        X_ = X.copy()
        X_[:, j] = np.mod(X[:, j]+shift, n_values[j])
        logden_X_ = log_den(X_, *args)
        if average:
            logden_X_ = np.mean(logden_X_, axis=0)
        ratio = np.exp(logden_X_ - logden_X)
        S[:, j] = ratio - 1.
    # print(np.unique(S))
    return S


def continuous_score(log_den, X, *args, average=False):
    """Score function for continuous-valued X"""
    def log_den_(X):
        if average:
            return np.mean(log_den(X, *args), axis=0)
        return log_den(X, *args)
    g = autograd.elementwise_grad(log_den_)
    G = g(X)
    return G


class UnnormalizedDensity(kgofden.UnnormalizedDensity):
    """
    An abstract class of a unnormalized probability density function.
    """

    @property
    @abstractmethod
    def var_type_disc(self):
        """Indicates if the observed variable is discrete
        or not. If True, it is discrete.
        """
        pass

    def score(self, X):
        """
        Return the score function evaluated at X. 
        Args:
            -X: an n x d numpy array
        Return:
            an n x d numpy array
        """
        if self.var_type_disc:
            return discrete_score(self.log_den,
                                  X, self.n_values
                                  )
        else:
            return self.grad_log(X)

    def log_grad(self, X):
        if not self.var_type_disc:
            raise NotImplementedError()

        grad = self.grad_den(X)
        sign = np.sign(grad)
        return np.log(sign*grad), sign


class Normal(kgofden.Normal, UnnormalizedDensity):

    var_type_disc = False

    def __init__(self, mean, cov):
        super(Normal, self).__init__(mean, cov)
        self.cov_evals = None


class IsotropicNormal(kgofden.IsotropicNormal, UnnormalizedDensity):
    """
    NormalizedDensity Implementation of
    isotropic normal distribution. 
    """
    var_type_disc = False

    def __init__(self, mean, variance):
        """
        mean: a numpy array of length d for the mean 
        variance: a positive floating-point number for the variance.
        """
        super(IsotropicNormal, self).__init__(mean, variance)
    

class MixtureDensity(UnnormalizedDensity):

    def __init__(self, densities, pmix=None):
        self.num_den = len(densities)
        self.densities = densities
        self.var_type_disc = densities[0].var_type_disc
        if self.var_type_disc:
            self.n_values = densities[0].n_values

        if pmix is None:
            self.pmix = None
        else:
            self.pmix = np.array(pmix)
            assert self.num_den == self.pmix.shape[0]
            if np.abs(np.sum(self.pmix) - 1) > 1e-8:
                raise ValueError('Mixture weights do not sum to 1.')

    def dim(self):
        return self.densities[0].dim()

    def get_datasource(self):
        data_sources = [p.get_datasource() for 
                        p in self.densities]
        return data.DSMixtureDist(data_sources, self.pmix)

    # TODO implement logsumexp (if the gradient is more stable
    # with own implementation)
    def log_normalized_den(self, X):
        n, _ = X.shape
        pmix = self.pmix
        expsum = np.zeros([n])
        # this is not memory efficient
        # logpmax = np.max(
        #     np.array([p.log_normalized_den(X) for p in self.densities]),
        #     axis=0
        # )
        logpmax = np.ones([n]) * np.NINF
        pmix_sum = 0.0
        for p in self.densities:
            tmp = np.array(p.log_normalized_den(X))
            logpmax = np.maximum(tmp, logpmax)
        for i, p in enumerate(self.densities):
            logp = p.log_normalized_den(X)
            if pmix is None:
                expsum = expsum + (np.exp(logp-logpmax)-expsum)/(i+1)
            else:
                pmix_sum += pmix[i]
                w = pmix[i] / pmix_sum
                expsum = expsum + w*(np.exp(logp-logpmax)-expsum)
        return np.log(expsum) + logpmax
    
    def log_den(self, X):
        return self.log_normalized_den(X)


class Categorical(UnnormalizedDensity, DiscreteFunction):

    var_type_disc = True

    def __init__(self, probs):
        self.probs = probs
        self._n_values = probs.shape[1]

    def dim(self):
        return self.probs.shape[0]

    def get_datasource(self):
        return data.DSCategorical(probs=self.probs)

    def log_normalized_den(self, X):
        n, d = X.shape
        probs = self.probs
        log_den = np.empty([n, d])
        for j in range(d):
            log_den[:, j] = np.log(probs[j, X[:, j]])
        return np.sum(log_den, axis=1)

    def log_den(self, X):
        return self.log_normalized_den(X)


class MultivariateBern(UnnormalizedDensity, DiscreteFunction):
    """
    Normalized Density class representing a dimensionality-wise independent
    multivariate Bernoulli distribution.
    """

    var_type_disc = True

    def __init__(self, probs):
        """
        probs = (d, ) numpy array
        """
        unit_interval_check(probs)
        self.probs = probs
        self._n_values = 2

    def dim(self):
        return len(self.probs)

    def get_datasource(self):
        return data.DSMultivariateBern(self.probs)

    def log_normalized_den(self, X):
        """
        X = binary (n, ) numpy array
        """
        _, d = X.shape
        P = np.vstack([1. - self.probs, self.probs]).T
        return np.sum(np.log(P[np.arange(d), X]), axis=1)

    def log_den(self, X):
        return self.log_normalized_den(X)


class Binomial(UnnormalizedDensity, DiscreteFunction):
    """
    NormalizedDensity Binomial distribution object.
    Allows a multivariate Bernoulli (n_k, p_k) 
    for k = 1 to d, where n_k is the number of trials
    and p_k is the success probability for kth 
    component. 

    Attributes:
        - n_trials (d,) numpy array, number of trials
        - probs: (d,) numpy array
    
    """
    var_type_disc = True

    def __init__(self, n_trials, probs):
        assert len(n_trials) == len(probs)
        unit_interval_check(probs)
        self.n_trials = n_trials
        self._n_values = n_trials + 1  # n_trials + 1
        self.probs = probs

    def dim(self):
        return len(self.probs)

    def get_datasource(self):
        return data.DSMultivariateBinomial(self.n_trials, self.probs)

    def log_normalized_den(self, X):
        """
        Args:
            - X = (n, d) numpy array 
        Returns:
            - (n, ) numpy array
        """
        log_probs = stats.binom.logpmf(X, n=self.n_trials,
                                       p=self.probs.reshape(-1))
        P = np.sum(log_probs, axis=1)
        return P

    def log_den(self, X):
        return self.log_normalized_den(X)


class BetaBinomSinglePriorMarginal(UnnormalizedDensity, DiscreteFunction):
    """
    NormalizedDensity object for the marginal of BetaBinomSinglePrior.
    Likelihood is a multivariate Bernoulli (n_k, p_k) 
    for k = 1 to d, where n_k is the number of trials
    and p_k is the success probability for kth 
    component. 

    Attributes:
        - n_trials (d,) numpy array = number of trials
        - probs: (d,) numpy array
    
    """
    var_type_disc = True 

    def __init__(self, n_trials, alpha, beta):
        self._n_values = n_trials + 1  # can take 0...n_k
        self.n_trials = n_trials
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError('alpha or beta is not positive')

    def eval_den(self, X):
        """
        Args:
            - X = (n, d) numpy array 
        Returns:
            - (n, ) numpy array
        """
 
        n_trials = self.n_trials 
        X_colsum = np.sum(X, axis=1)
        total = np.sum(n_trials)
        C = np.prod(special.binom(n_trials, X), axis=1)
        den = C * special.beta(X_colsum+self.alpha, total-X_colsum+self.beta)
        den /= special.beta(self.alpha, self.beta)
        return den

    def dim(self):
        return len(self._n_values)

    def log_normalized_den(self, X):
        n_trials = self.n_trials 
        C = np.sum(np.log(special.binom(n_trials, X)), axis=1)
        log_den = C + np.sum(np.log(special.beta(X+self.alpha,
                                                 n_trials-X+self.beta)), axis=1)

        log_den -= np.log(special.beta(self.alpha, self.beta))
        return log_den
        # return np.log(self.eval_den(X))
    
    def log_den(self, X):
        return self.log_normalized_den(X)
        

class SigmoidBeliefNet(UnnormalizedDensity, DiscreteFunction):
    """ 
    Normalized Density object for Sigmoid Belief Network.

    Attributes:
        - W: d x dh numpy array for weight
        upper triangular matrix with diagonal zero
        - h: numpy array of size [dh,] for hidden state
        - b: numpy array of size [d,] for bias
    """
    var_type_disc = True

    def __init__(self, W, h, b=None):
        self._n_values = 2
        d = W.shape[0]
        self.W = W
        self.h = h
        self.b = b if b is not None else np.zeros(d)
        assert len(self.h) == W.shape[1]
        assert len(self.b) == d

    def log_normalized_den(self, X):
        W = self.W
        b = self.b
        h = self.h
        exponent = (np.dot(W, h)+b)
        probs = special.expit(exponent)
        return np.sum(X*np.log(probs)+(1.-X)*(np.log(1-probs)), axis=1)

    def dim(self):
        return self.W.shape[0]

    def get_datasource(self):
        return data.DSSigmoidBeliefNet(W=self.W, h=self.h, b=self.b)

    def log_den(self, X):
        return self.log_normalized_den(X)
