"""
Module containing implementations of various
model comparison tests for latent variable models.
"""
__author__ = 'noukoudashisoup'

import lkgof
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
        Defaults to util.second_order_ustat_variance_jackknife.
    """

    def __init__(self, p, q, k, l, seed=11, alpha=0.05,
                 varest=util.second_order_ustat_variance_jackknife,
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
        Defaults to util.second_order_ustat_variance_jackknife
    """

    def __init__(self, modelp, modelq, k, l,
                 mc_param_p, mc_param_q,
                 ps_p=None, ps_q=None,
                 alpha=0.01, seed=11,
                 varest=util.second_order_ustat_variance_jackknife,):
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
                'var: {}, scale: {}, threshold: {}'.format(
                    var, scale, scale*stats.norm.ppf(1-alpha))
            )
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

# end class LDC_KSD

class SC_MMD(object):
    """
    A test for model comparison using the Maximum Mean Discrepancy (MMD)
    proposed by Bounliphone, et al 2016 (ICLR)
    """

    def __init__(self, datap, dataq, k, alpha=0.01):
        """
        :param datap: a lkgof.data.Data object representing an i.i.d. sample X
            (from model 1)
        :param dataq: a lkgof.data.Data object representing an i.i.d. sample Y
            (from model 2)
        :param k: a lkgof.Kernel
        :param alpha: significance level of the test
        """
        self.datap = datap
        self.dataq = dataq
        self.alpha = alpha
        self.k = k

    def perform_test(self, dat):
        """perform the model comparison test and return values computed in a
        dictionary: 
        {
            alpha: 0.01,
            pvalue: 0.0002,
            test_stat: 2.3,
            h0_rejected: True,
            time_secs: ...
        }

        :param dat: an instance of lkgof.data.Data
        """
        with util.ContextTimer() as t:
            alpha = self.alpha
            X = self.datap.data() 
            Y = self.dataq.data() 
            Z = dat.data()
            nx = X.shape[0]
            ny = Y.shape[0]
            nz = Z.shape[0]
            n = (nx + ny + nz)

            # mean is not yet scaled by \sqrt{n}
            # The variance is the same for both H0 and H1.
            mean, var = self.get_H1_mean_variance(dat)
            if not util.is_real_num(var) or var < 0:
                log.l().warning('Invalid H0 variance. Was {}'.format(var))
            stat = (n**0.5) * mean
            # Assume the mean of the null distribution is 0
            pval = stats.norm.sf(stat, loc=0, scale=var**0.5)
            rejected = (stat > stats.norm.ppf(1-alpha)*(var**0.5))

            # rejected = (mean > stats.norm.ppf(1-alpha)*(var**0.5))
            if not util.is_real_num(pval):
                log.l().warning('p-value is not a real number. Was {}'.format(pval))


        results = {
            'alpha': self.alpha, 'pvalue': pval, 'test_stat': stat,
            'h0_rejected': rejected, 'time_secs': t.secs,
        }
        return results

    def compute_stat(self, dat):
        """
        Compute the test statistic (difference between MMD estimates)
        :returns: the test statistic (a floating-point number)
        """
        mean_h1 = self.get_H1_mean_variance(dat, return_variance=False)
        n = dat.sample_size()
        return (n**0.5) * mean_h1


    def get_H1_mean_variance(self, dat, return_variance=True):
        """
        Return the (scaled) mean and variance under H1 of the 
        test statistic = 
            sqrt(n)*(MMD_u(Z_{n_z}, X_{n_x})^2 - MMD_u(Z_{n_z}, Y_{n_y})^2)^2.
        The estimator of the mean is unbiased (can be negative). 
        Note that the mean is divided by n**0.5

        :returns: (mean, variance)
        """
        # form a two-sample test dataset between datap and dat (data from R)
        Z = dat.data()
        X = self.datap.data()
        Y = self.dataq.data()
        nx = X.shape[0]
        ny = X.shape[0]
        nz = Z.shape[0]

        n = nx + ny + nz
        k = self.k

        diagidx_x = np.diag_indices(nx)
        diagidx_y = np.diag_indices(ny)
        Kxx = k.eval(X, X)
        Kxx[diagidx_x] = 0. # discard diagonal
        Kyy = k.eval(Y, Y)
        Kyy[diagidx_y] = 0. # discard diagonal
        Kxz = k.eval(X, Z)
        Kyz = k.eval(Y, Z)

        # Kzz term is not required for computing the MMD difference
        mmd_mean_pr = (np.sum(Kxx)/(nx*(nx-1)) - 2. * Kxz.mean())
        mmd_mean_qr = (np.sum(Kyy)/(ny*(ny-1)) - 2. * Kyz.mean())

        mean_h1 = mmd_mean_pr - mmd_mean_qr
        # This always return a variance. But will be None if is_var_computed=False
        if not return_variance:
            return mean_h1

        x_mean_emb = np.sum(Kxx, axis=1, keepdims=True) / (nx - 1)
        z_meanemb_at_x = np.mean(Kxz, axis=1, keepdims=True)
        diff_x = x_mean_emb - z_meanemb_at_x
        var_x_tmp = (diff_x - diff_x.T)**2
        var_x = ( np.sum(var_x_tmp) - np.trace(var_x_tmp) ) / ( 2. * nx * (nx-1) )
        var_x = 4. * (n/nx) * var_x


        y_mean_emb = np.sum(Kyy, axis=1, keepdims=True) / (ny - 1)
        z_meanemb_at_y = np.mean(Kyz, axis=1, keepdims=True)
        diff_y = y_mean_emb - z_meanemb_at_y
        var_y_tmp = (diff_y - diff_y.T)**2
        var_y = ( np.sum(var_y_tmp) - np.trace(var_y_tmp) ) / ( 2. * ny * (ny-1) )
        var_y = 4. * (n/ny) * var_y

        
        x_meanemb_at_z = np.mean(Kxz, axis=0, keepdims=True)
        y_meanemb_at_z = np.mean(Kyz, axis=0, keepdims=True)
        diff_z = x_meanemb_at_z - y_meanemb_at_z
        var_z_tmp = (diff_z.T - diff_z)**2
        var_z = (np.sum(var_z_tmp) - np.trace(var_z_tmp)) / ( 2. * nz * (nz-1) )
        var_z = 4. * (n/nz) * var_z

        var_h1 = var_x + var_y + var_z
        return mean_h1, var_h1

    @staticmethod
    def ustat_H1_variance_estimator(Kxx, Kxz, Kyy, Kyz):
        Kxx = Kxx.copy()
        nx = Kxx.shape[0]
        Kxx[np.diag_indices(nx)] = 0. # discard diagonal
        Kyy = Kyy.copy() 
        ny = Kyy.shape[0]
        Kyy[np.diag_indices(ny)] = 0. # discard diagonal
        nz = Kxz.shape[1]
        assert nx == Kxx.shape[1]
        assert ny == Kyz.shape[0]
        assert nz == Kyz.shape[1]
        n = nx + ny + nz
        
        lf = lkgof.util.lower_factorial
        kxx_row_norm = np.sum( np.sum(Kxx, axis=0)**2 )
        kxx_frob_norm = np.sum(Kxx**2)
        kxx_sum_sq = np.sum(Kxx)**2
        var_x = (kxx_row_norm-kxx_frob_norm)/lf(nx, 3)
        var_x -= (kxx_sum_sq-4*kxx_row_norm+2*kxx_frob_norm) / lf(nx, 4)
        # kxz var wrt x
        kxz_frob_sq = np.sum(Kxz**2)
        kxz_xmatp_sum = np.sum(Kxz.T @ Kxz)
        var_x += (kxz_xmatp_sum - kxz_frob_sq) / (nx * lf(nz, 2))
        kxz_sum_sq = np.sum(Kxz) ** 2
        kxz_zmatp_sum = np.sum(Kxz @ Kxz.T)
        kxz_sq_est = (kxz_sum_sq+kxz_frob_sq-kxz_xmatp_sum-kxz_zmatp_sum) / (nx*nz*(nx-1)*(nz-1))
        var_x -= kxz_sq_est
        # subtract cov 
        var_x -= 2.* np.sum(Kxx @ Kxz) / (nz*nx*(nx-1))
        var_x += 2.* (np.sum(Kxx) * np.sum(Kxz) - 2*np.sum(Kxx@Kxz)) / (nz*lf(nx, 3))

        kyy_row_norm = np.sum( np.sum(Kyy, axis=0)**2 )
        kyy_frob_norm = np.sum(Kyy**2)
        kyy_sum_sq = np.sum(Kyy)**2
        var_y = (kyy_row_norm-kyy_frob_norm)/lf(ny, 3)
        var_y -= (kyy_sum_sq-4*kyy_row_norm+2*kyy_frob_norm) / lf(ny, 4)
        kyz_frob_sq = np.sum(Kyz**2)
        var_y += (np.sum(Kyz.T@Kyz) - kyz_frob_sq) / (ny * lf(nz, 2))
        kyz_sum_sq = np.sum(Kyz) ** 2
        kyz_zmatp_sum = np.sum(Kyz @ Kyz.T)
        kyz_ymatp_sum = np.sum(Kyz.T @ Kyz)
        kyz_sq_est = (kyz_sum_sq+kyz_frob_sq-kyz_ymatp_sum-kyz_zmatp_sum) / (ny*nz*(ny-1)*(nz-1))
        var_y -= kyz_sq_est
        # subtract cov 
        var_y -= 2.* np.sum(Kyy * Kyz.mean(axis=1)) / lf(ny, 2)
        var_y += 2.* (np.sum(Kyy) * np.sum(Kyz) - 2*np.sum(Kyy@Kyz)) / (nz*lf(ny, 3))
        var_z = (kxz_zmatp_sum - kxz_frob_sq) / (nz*nx*(nx-1))
        var_z -= kxz_sq_est
        var_z += (kyz_zmatp_sum - kyz_frob_sq)/ (nz*ny*(ny-1))
        var_z -= kyz_sq_est
        # subtract cov
        tmp_mat = Kxz.sum(axis=0, keepdims=True).T @ Kyz.sum(axis=0, keepdims=True)
        var_z -= 2. * np.sum(Kxz@Kyz.T)/(nx*ny*nz) 
        var_z += 2. * (np.sum(tmp_mat) - np.sum(np.trace(tmp_mat))) / (nx*ny*lf(nz, 2))

        var_x = 4. * (n/nx) * var_x
        var_y = 4. * (n/ny) * var_y
        var_z = 4. * (n/nz) * var_z
        return var_x + var_y + var_z
    
    @staticmethod
    def median_heuristic_bounliphone(X, Y, Z, subsample=1000, seed=287):
        """
        Return the median heuristic as implemented in 
        https://github.com/wbounliphone/relative_similarity_test/blob/4884786aa3fe0f41b3ee76c9587de535a6294aee/relativeSimilarityTest_finalversion.m

            % selection of theBandwidth;
            myX = pdist2(X,Y);
            myX = myX(:);
            theBandwidth(1) = sqrt(median(myX(:))/2);
            myX = pdist2(X,Z);
            myX = myX(:);
            theBandwidth(2) = sqrt(median(myX(:))/2);
            theBandwidth=mean(theBandwidth);
            params.sig=theBandwidth;
            localSig=params.sig;

        The existence of sqrt(..) above does not make sense. Probably they
        thought pdist2 returns squared Euclidean distances.  In fact, it appears
        to return just Euclidean distances. Having sqrt(..) above would lead to
        the use of square root of Euclidean distances.
        The computation in the code above is for v (Gaussian width) where the
        Gaussian kernel is exp(-|x-y|^2/v^2) (no factor of 2 in the denominator).

        We translate the above code into our parameterization 
        exp(-|x-y|^2/(2*s2)) where s is the squared Gaussian width.
        We implement the following
        code by keeping the sqrt above, and assuming that pdist2(...) returns
        squared Euclidean distances. So,

        s2 = 0.5*mean([median(squared_pdist(Y, Z))**0.5, median(squared_pdist(X,Z))**0.5 ])**2

        * X, Y: samples from two models.
        * Z: reference sample 
        """
        # subsample first
        nx = X.shape[0]
        ny = Y.shape[0]
        nz = Z.shape[0]
        if nx != ny:
            raise ValueError('X and Y do not have the same sample size. nx={}, ny={}'.format(nx, ny))
        if ny != nz:
            raise ValueError('Y and Z do not have the same sample size. ny={}, nz={}'.format(ny, nz))
        n = nx
        assert subsample > 0
        with util.NumpySeedContext(seed=seed):
            ind = np.random.choice(n, min(subsample, n), replace=False)
            X = X[ind, :]
            Y = Y[ind, :]
            Z = Z[ind, :]

        sq_pdist_yz = util.dist_matrix(Y, Z)**2
        med_yz = np.median(sq_pdist_yz)**0.5

        sq_pdist_xz = util.dist_matrix(X, Z)**2
        med_xz = np.median(sq_pdist_xz)**0.5
        sigma2 = 0.5*np.mean([med_yz, med_xz])**2
        return sigma2


# end of class SC_MMD
