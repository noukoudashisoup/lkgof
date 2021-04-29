"""
Module containing implementations of various
model comparison tests for latent variable models.
"""
__author__ = 'noukoudashisoup'

import lkgof.util as util
from lkgof import goftest as gof
from lkgof import log
from lkgof.goftest import KernelSteinTest
from lkgof.goftest import LatentKernelSteinTest
import autograd.numpy as np
from scipy import stats


class DC_KSD(object):
    """
    KSD model comparison test for unnormalized
    models with tractable marginals.
    
    Args:
    - p:
        lkgof.density.UnnormalizedDensity object
    - q:
        lkgof.density.UnnormalizedDensity object
    - k: kernel object
    - l: kernel object
    - alpha: significance level
    - seed: random seed
    - varest:
        Variance estimator method.
        Defaults to util.second_order_ustat_variance_ustat.
    """

    def __init__(self, p, q, k, l, seed=11, alpha=0.05,
                 varest=util.second_order_ustat_variance_ustat,
                 ):
        self.p = p
        self.q = q
        self.alpha = alpha
        self.k = k
        self.l = l
        self.varest = varest

    def perform_test(self, dat):
        """
        :param dat: an instance of lkgof.data.Data
        """
        with util.ContextTimer() as t:
            alpha = self.alpha
            X = dat.data()
            n = X.shape[0]
            # mean and variance are not yet scaled by \sqrt{n}
            mean, var = self.get_mean_variance(dat)

            # var = np.max([var, 0.])
            var = np.sqrt(var**2)
            stat = (n**0.5) * mean
            scale = (var * n)**0.5
            log.l().info(
                'scale: {}, threshold: {}'.format(
                    scale, scale*stats.norm.ppf(1-alpha))
            )
            if scale <= 1e-7:
                (('SD of the null distribution is too small.'
                  'Was {}. Will not reject H0.').format(scale))
                pval = np.inf
            else:
                # Assume the mean of the null distribution is 0
                pval = stats.norm.sf(stat, loc=0, scale=scale)

        rejected = (mean > stats.norm.ppf(1-alpha)*(var**0.5))
        results = {'alpha': self.alpha, 'pvalue': pval, 'test_stat': stat,
                   'h0_rejected': rejected, 'time_secs': t.secs, }
        return results

    def compute_stat(self, dat):
        X = dat.data()
        n, d = X.shape
        p = self.p
        q = self.q
        ksd_pr = KernelSteinTest.ustat_h1_mean_variance(
            X, p, self.k, return_variance=False)
        ksd_qr = KernelSteinTest.ustat_h1_mean_variance(
            X, q, self.l, return_variance=False)
        stat = (n**0.5) * (ksd_pr - ksd_qr)
        return stat

    def get_mean_variance(self, dat):
        X = dat.data()
        n, d = X.shape
        k = self.k
        l = self.l
        p = self.p
        q = self.q

        statp = KernelSteinTest.ustat_h1_mean_variance(
            X, p, k, return_variance=False, use_unbiased=True)
        statq = KernelSteinTest.ustat_h1_mean_variance(
            X, q, l, return_variance=False, use_unbiased=True)
        mean_h1 = (statp - statq)
        log.l().info('diff = {}-{}'.format(statp, statq))

        Hp = KernelSteinTest.stein_kernel_gram(X, p, k)
        Hq = KernelSteinTest.stein_kernel_gram(X, q, l)
        var_h1 = self.varest(Hp-Hq)
        if var_h1 <= 0:
            log.l().warning('var is not positive. Was {}'.format(var_h1))

        return mean_h1, var_h1


class LDC_KSD(object):
    """
    KSD model comparison test for latent variable
    models.
    
    Args:
    - modelp: model.LatentVariableModel
    - modelq: model.LatentVariableModel
    - k: kernel object
    - l: kernel object
    - mc_param_p:
        goftest.MCParam. Parameters required
        for Monte Carlo approximation
    - mc_param_q:
        goftest.MCParam. Parameters required
        for Monte Carlo approximation
    - ps_p:
        posterior sampler for p. Defaults to None.
    - ps_q:
        posterior sampler for p. Defaults to None.
    - alpha: significance level
    - seed: random seed
    - varest:
        Variance estimator method.
        Defaults to util.second_order_ustat_variance_ustat.
    """

    def __init__(self, modelp, modelq, k, l,
                 mc_param_p, mc_param_q,
                 ps_p=None, ps_q=None,
                 alpha=0.01, seed=11,
                 varest=util.second_order_ustat_variance_ustat):
        self.modelp = modelp
        self.modelq = modelq
        self.k = k
        self.l = l
        self.alpha = alpha
        self.seed = seed
        self.mc_param_p = mc_param_p
        self.mc_param_q = mc_param_q
        self.ps_p = (modelp.posterior if ps_p is None
                     else ps_p)
        self.ps_q = (modelq.posterior if ps_q is None
                     else ps_q)
        self.varest = varest

    def perform_test(self, dat):
        """
        :param dat: an instance of lkgof.data.Data
        """
        with util.ContextTimer() as t:
            alpha = self.alpha
            X = dat.data()
            n = X.shape[0]
            # mean and variance are not yet scaled by \sqrt{n}
            mean, var = self.get_mean_variance(dat)

            var = np.max([var, 0.])
            stat = (n**0.5) * mean
            scale = (var * n)**0.5
            log.l().info(
                'scale: {}, threshold: {}'.format(
                    scale, scale*stats.norm.ppf(1-alpha))
            )
            if scale <= 1e-7:
                (('SD of the null distribution is too small.'
                  'Was {}. Will not reject H0.').format(scale))
                pval = np.inf
            else:
                # Assume the mean of the null distribution is 0
                pval = stats.norm.sf(stat, loc=0, scale=scale)
            rejected = (mean > stats.norm.ppf(1-alpha)*(var**0.5))

        results = {'alpha': self.alpha, 'pvalue': pval, 'test_stat': stat,
                   'h0_rejected': rejected, 'time_secs': t.secs, }
        return results

    def compute_stat(self, dat):
        X = dat.data()
        n, d = X.shape
        p = self.modelp
        q = self.modelq
        ps_p, n_sample_p, n_burnin_p = self.mc_param_p
        ps_q, n_sample_q, n_burnin_q = self.mc_param_q
        ksd_pr = LatentKernelSteinTest.ustat_h1_mean_variance(
            X, p, self.k, ps_p, return_variance=False,
            seed=self.seed+1, n_sample=n_sample_p,
            n_burnin=n_burnin_p,
        )

        ksd_qr = LatentKernelSteinTest.ustat_h1_mean_variance(
            X, q, self.k, ps_q, return_variance=False,
            seed=self.seed+2, n_sample=n_sample_p,
            n_burnin=n_burnin_q,
        )
        stat = (n**0.5) * (ksd_pr - ksd_qr)
        return stat

    def get_mean_variance(self, dat):
        """Returns esimates of the KSD difference and 
        its variance"""
        
        assert isinstance(self, LDC_KSD)
        X = dat.data()
        n, d = X.shape
        k = self.k
        l = self.l
        seed = self.seed
        modelp = self.modelp
        modelq = self.modelq
        ps_p = self.ps_p
        ps_q = self.ps_q
        n_sample_p, n_burnin_p = self.mc_param_p
        n_sample_q, n_burnin_q = self.mc_param_q

        statp, Hp = LatentKernelSteinTest.ustat_h1_mean_variance(
            X, modelp, k, ps_p, return_variance=False, use_unbiased=True,
            return_gram=True,
            seed=seed+1, n_burnin=n_burnin_p,
            n_sample=n_sample_p
        )
        statq, Hq = gof.LatentKernelSteinTest.ustat_h1_mean_variance(
            X, modelq, l, ps_q, return_variance=False, use_unbiased=True,
            return_gram=True,
            seed=seed+2, n_burnin=n_burnin_q,
            n_sample=n_sample_q,
        )
        mean_h1 = (statp - statq)
        # log.l().info('diff = {}-{}'.format(statp, statq))
        var_h1 = self.varest(Hp-Hq)

        return mean_h1, var_h1
