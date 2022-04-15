"""Module containing goodness-of-fit tests"""

from kgof.goftest import *
from kgof import goftest as gof
from lkgof import model
import lkgof.util as util
import numpy as np
from typing import NamedTuple
from lkgof.stein import stein_kernel_gram

class KernelSteinTest(gof.KernelSteinTest):

    @staticmethod
    def ustat_h1_mean_variance(X, p, k, return_variance=True,
                               use_unbiased=True):
        """
        Returns mean and variance of KSD^2
        """
        n, d = X.shape
        H = KernelSteinTest.stein_kernel_gram(X, p, k)
        if use_unbiased:
            mean = (np.sum(H)-np.sum(np.diag(H))) / (n*(n-1))
        else:
            mean = np.mean(H)

        if not return_variance:
            return mean

        variance = util.second_order_ustat_variance_ustat(H)
        
        return mean, variance

    @staticmethod
    def stein_kernel_gram(X, p, k):
        """
        Compute the Stein kernel gram matrix hp(x_i, x_j)
        Args: 
            - X: an n x d data numpy array
            - p: a lkgof.density object
            - k: a KST/DKST object
        Return:
            - an n x n array
        """
        n, d = X.shape
        # n x d matrix of gradients
        score_p = p.score(X)
        return stein_kernel_gram(X, score_p, k)

    def compute_stat(self, dat, return_ustat_gram=False):
        """
        Compute the U statistic as in Section 4 of Liu et al.., 2016.
        return_ustat_gram: If True, then return the n x n matrix used to
            compute the statistic (by taking the mean of all the elements)
        """

        X = dat.data()
        n, d = X.shape
        p = self.p
        k = self.k
        H = self.stein_kernel_gram(X, p, k)
        # U-statistic
        stat = 1. / (n-1) * (np.sum(H)-np.sum(np.diag(H)))
        if return_ustat_gram:
            return stat, H
        else:
            return stat


class LKSDH0SimNaiveBoot(gof.H0Simulator):
    """
    An asymptotic null distribution simulator for LKSD. 
    Simulate by sampling data from the model and computing 
    the test statistic. 
    """

    def __init__(self, n_simulate, seed):
        assert n_simulate > 0
        self.n_simulate = n_simulate
        self.seed = seed

    def simulate(self, gof, dat):
        n_simulate = self.n_simulate
        sim_stats = np.zeros(n_simulate)
        ds = gof.model.get_datasource()
        for sim_i in range(n_simulate):
            sim_seed = self.seed + sim_i + 1
            if gof.fix_z:
                p = gof.p
            else:
                p = gof.model.get_empirical_density(gof.nz, sim_seed)
            sim_dat = ds.sample(dat.n(), sim_seed)
            sim_stats[sim_i] = gof.compute_stat(sim_dat, p,
                                                return_ustat_gram=None)
 
        return {'sim_stats': sim_stats}


def eval_score(X, lvm,
               post_sampler,
               seed=13, n_burnin=500,
               n_sample=500):
    assert isinstance(lvm, model.LatentVariableModel)
    latents = post_sampler(X, n_sample,
                           seed, n_burnin=n_burnin,
                           )
    score_p = lvm.score_joint(X, latents)
    return score_p

class MCParam(NamedTuple):
    """A class representing parameters
    required for Markov chain samplers"""
    n_sample: int = 500
    n_burnin: int = 300


class LatentKernelSteinTest(gof.GofTest):
    """
    Goodness-of-fit test for latent variable
    models using approximate kernelized Stein discrepancy.
    """

    def __init__(self, lvm, k, mc_param,
                 alpha=0.01, ps=None, n_simulate=500,
                 seed=11, bootstrapper=None):
        """
        p: lkgof.LatentVariableModel
        """
        assert type(mc_param) is MCParam
        self.lvm = lvm
        self.k = k
        self.alpha = alpha
        self.n_simulate = n_simulate
        self.seed = seed
        self.ps = (lvm.posterior if ps is None 
                   else ps)
        self.mc_param = mc_param
        
        if bootstrapper is None:
            self.bootstrapper = gof.bootstrapper_multinomial
        else:
            self.bootstrapper = bootstrapper

    def perform_test(self, dat, return_simulated_stats=False,
                     return_ustat_gram=False):
        """
        dat: a instance of Data
        """
        with util.ContextTimer() as t:
            alpha = self.alpha
            n_simulate = self.n_simulate
            X = dat.data()
            n = X.shape[0]

            _, H = self.compute_stat(dat, return_ustat_gram=True)
            test_stat = (np.sum(H) - np.sum(np.diag(H))) / (n-1)
            # bootrapping
            sim_stats = np.zeros(n_simulate)
            with util.NumpySeedContext(seed=self.seed):
                for i in range(n_simulate):
                    W = np.random.multinomial(n, (np.ones(n)/n))
                    Wt = (W-1.0)/n
                    # n * [ (1/n^2) * \sum_i \sum_j h(x_i, x_j) w_i w_j ]
                    boot_stat = n * ( Wt.dot(H.dot(Wt)) - np.diag(H).dot(Wt**2) )
                    # This is a bootstrap version of n*U_n
                    sim_stats[i] = boot_stat
 
            # approximate p-value with the permutations 
            pvalue = np.mean(sim_stats > test_stat)
 
        results = {'alpha': self.alpha, 'pvalue': pvalue,
                   'test_stat': test_stat,
                   'h0_rejected': pvalue < alpha,
                   'n_simulate': n_simulate,
                   'time_secs': t.secs,
                   }
        if return_simulated_stats:
            results['sim_stats'] = sim_stats
        if return_ustat_gram:
            results['H'] = H
            
        return results

    def compute_stat(self, dat, return_ustat_gram=False):
        """
        Compute the U statistic as in Section 4 of Liu et al.., 2016.
        return_ustat_gram: If True, then return the n x n matrix used to
            compute the statistic (by taking the mean of all the elements)
        """

        X = dat.data()
        lvm = self.lvm
        k = self.k
        ps = self.ps
        n_sample, n_burnin = self.mc_param

        return self.ustat_h1_mean_variance(X, lvm, k, ps,
                                           return_variance=False,
                                           return_gram=return_ustat_gram,
                                           seed=self.seed,
                                           n_burnin=n_burnin,
                                           n_sample=n_sample,
                                           )


    @staticmethod
    def ustat_h1_mean_variance(X, lvm, k, post_sampler,
                               return_variance=True,
                               return_gram=False,
                               use_unbiased=True,
                               seed=13,
                               n_burnin=300,
                               n_sample=500,
                               ):
        """
        Returns an estimate of KSD^2 and its variance.

        Args: 
            - X: numpy.ndarray of size n x d
            - lvm: lkgof.model.LatentVariableModel
            - k: kernel object
            - post_sampler:
                sampler for the latent variables of lvm
            - return_variance:
                if True, return the variance of
                the U-stat
            - return_gram:
                if True, return the KSD gram matrix used
                to compute the stat
            - use_unbiased:
                if True, compute U-stat
            - seed:
            - n_burnin:
                burn-in step size for post_sampler
            - n_sample:
                the number of sample points returned by
                post_sampler
            
        Returns:
        Returns a tuple either of 
            (ustat, ) if return_variance is True,
            (ustat, ustat_variance) if return_variance is False
        Append the gram matrix used to the tuple
        above if return_gram is True.
        """

        def add_gram(*args, return_gram=False):
            if return_gram:
                return args + (H, )
            if len(args) == 1:
                return args[0]
            return args

        n, d = X.shape
        assert isinstance(lvm, model.LatentVariableModel)

        score_p = eval_score(X, lvm,
                             post_sampler, seed,
                             n_burnin=n_burnin,
                             n_sample=n_sample,
                             )
        H = stein_kernel_gram(X, score_p, k)

        if use_unbiased:
            mean = (np.sum(H)-np.sum(np.diag(H))) / (n*(n-1))
        else:
            mean = np.mean(H)

        if not return_variance:
            return add_gram(mean, return_gram=return_gram)

        variance = util.second_order_ustat_variance_ustat(H)
        return add_gram(mean, variance, return_gram=return_gram)

# end LatentKernelSteinTest
