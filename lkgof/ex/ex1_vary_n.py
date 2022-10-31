"""Simulation to get the test power vs increasing sample size"""

__author__ = 'noukoudashisoup'

import os 
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

import lkgof.glo as glo
import lkgof.mctest as mct
import lkgof.model as model
import lkgof.data as data
# goodness-of-fit test
import lkgof.kernel as kernel
import lkgof.util as util
import lkgof.mcmc as mcmc
from lkgof.mctest import SC_MMD
from lkgof.goftest import MCParam

# need independent_jobs package 
# https://github.com/wittawatj/independent-jobs
# The independent_jobs and kgof have to be in the global search path (.bashrc)
import independent_jobs as inj
from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.SingleResult import SingleResult
from independent_jobs.aggregators.SingleResultAggregator import SingleResultAggregator
from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.tools.Log import logger
#import numpy as np
import autograd.numpy as np
import sys 

"""
All the method functions (starting with met_) return a dictionary with the
following keys:
    - test_result: the result from calling perform_test(te).
    - time_secs: run time in seconds 

    * A method function may return an empty dictionary {} if the inputs are not
    applicable. For example, if density functions are not available, but the
    method function is KSD which needs them.

All the method functions take the following mandatory inputs:
    - P: a lkgof.model.LatenVariableModel (candidate model 1)
    - Q: a lkgof.model.LatentVariableModel (candidate model 2)
    - data_source: a kgof.data.DataSource for generating the data (i.e., draws
          from R)
    - n: total sample size. Each method function should draw exactly the number
          of points from data_source.
    - r: repetition (trial) index. Drawing samples should make use of r to
          set the random seed.
    -------
    - A method function may have more arguments which have default values.
"""


def sample_pqr(P, Q, ds_r, n, r, only_from_r=False, 
               pqr_same_sample_size=False):
    """
    Generate three samples from the three data sources given a trial index r.
    All met_ functions should use this function to draw samples. This is to
    provide a uniform control of how samples are generated in each trial.

    ds_p: DataSource for model P
    ds_q: DataSource for model Q
    ds_r: DataSource for the data distribution R
    n: sample size to draw from each
    r: trial index

    Return (datp, datq, datr) where each is a Data containing n x d numpy array
    Return datr if only_from_r is True.
    """
    datr = ds_r.sample(n, seed=r+30)
    if only_from_r:
        return datr
    ds_p = P.get_datasource()
    ds_q = Q.get_datasource()
    if pqr_same_sample_size:
        n_model_samples = n
    else:
        n_burnin_p = burnin_sizes.get(type(P), 500)
        n_burnin_q = burnin_sizes.get(type(Q), 500)
        n_model_samples = n + max(n_burnin_p, n_burnin_q) + n_mcsamples
        if type(P) is model.PPCA:
            n_model_samples = n + 5000
    datp = ds_p.sample(n_model_samples, seed=r+1000)
    datq = ds_q.sample(n_model_samples, seed=r+2000)
    return datp, datq, datr

# -------------------------------------------------------


def _met_mmd(P, Q, data_source, n, r, k,):
    """
    Wrapper for MMD model comparison test (relative test). 
    Different tests can be defined depending on the input kernel k.
    """
    if not P.has_datasource() or not Q.has_datasource():
        # Not applicable. Return {}.
        return {}

    # Start the timer here
    with util.ContextTimer() as t:
        # sample some data 
        datp, datq, datr = sample_pqr(P, Q, data_source, n, r, only_from_r=False)

        X, Y, Z = datp.data(), datq.data(), datr.data()

        scmmd = SC_MMD(datp, datq, k, alpha=alpha)
        scmmd_result = scmmd.perform_test(datr)

    return {
            'test_result': scmmd_result, 'time_secs': t.secs}


def met_gmmd_med(P, Q, data_source, n, r):
    """
    Bounliphone et al., 2016's MMD-based 3-sample test.
    * Gaussian kernel. 
    * Gaussian width = median of the sample 
    """
    if not P.has_datasource() or not Q.has_datasource():
        # Not applicable. Return {}.
        return {}

    # sample some data 
    kernel_data = sample_pqr(None, None, data_source, n=kernel_datasize, r=0, only_from_r=True)
    X = kernel_data.data()

    # datp, datq, datr = sample_pqr(P, Q, data_source, n, r, only_from_r=False)
    # X, Y, Z = datp.data(), datq.data(), datr.data()

    # # hyperparameters of the test
    # medxz = util.meddistance(np.vstack((X, Z)), subsample=1000)
    # medyz = util.meddistance(np.vstack((Y, Z)), subsample=1000)
    # medxyz = np.mean([medxz, medyz])
    medX = util.meddistance(X)
    k = kernel.KGauss(sigma2=medX**2)
    scmmd_result = _met_mmd(P, Q, data_source, n, r, k=k,)
    return scmmd_result


def met_imqmmd_cov(P, Q, data_source, n, r):
    """
    Bounliphone et al., 2016's MMD-based 3-sample test.
    * Precondition IMQ kernel with sample covariance preconditioner
    """
    # sample some data 
    kernel_data = sample_pqr(None, None, data_source, n=kernel_datasize, r=0, only_from_r=True)
    X = kernel_data.data()

    cov = np.cov(X, rowvar=False)
    k = kernel.KPIMQ(P=cov)
    scmmd_result = _met_mmd(P, Q, data_source, n, r, k=k)
    return scmmd_result


def met_imqmmd_med(P, Q, data_source, n, r):
    """
    Bounliphone et al., 2016's MMD-based 3-sample test.
    * IMQ kernel with median scaling
    """
    # sample some data
    kernel_data = sample_pqr(None, None, data_source, n=kernel_datasize, r=0, only_from_r=True)
    X = kernel_data.data()
    d = X.shape[1]
    medX = util.meddistance(X)
    k = kernel.KPIMQ(medX**2 * np.eye(d))

    scmmd_result = _met_mmd(P, Q, data_source, n, r, k=k)
    return scmmd_result


def met_imqmmd_medtrunc(P, Q, data_source, n, r):
    """
    Bounliphone et al., 2016's MMD-based 3-sample test.
    * IMQ kernel with median scaling with truncation weight
    """
    # sample some data
    kernel_data = sample_pqr(None, None, data_source, n=kernel_datasize, r=0, only_from_r=True)
    X = kernel_data.data()
    d = X.shape[1]
    medX = util.meddistance(X)
    kimq = kernel.KPIMQ(np.eye(d)*medX**2)
    # kimq = kernel.KPIMQ(np.cov(X, rowvar=False))
    L = P.lim_lower * np.ones(d)
    U = P.lim_upper * np.ones(d)
    w = kernel.WeightSmoothIntevalProduct(L=L, U=U)
    kw = kernel.KSTWeight(w)
    k = kernel.KSTProduct(kimq, kw)

    scmmd_result = _met_mmd(P, Q, data_source, n, r, k=k)
    return scmmd_result


def _met_ksd(P, Q, data_source, n, r, k,
             varest=util.second_order_ustat_variance_jackknife,
             ):
    """
    Wrapper for KSD model comparison test (relative test). 
    Different tests can be defined depending on the input kernel k.
    """
    if not P.has_unnormalized_density() or not Q.has_unnormalized_density():
        # Not applicable. Return {}.
        return {}

    p = P.get_unnormalized_density()
    q = Q.get_unnormalized_density()
    # sample some data 
    datr = sample_pqr(None, None, data_source, n, r, only_from_r=True)

    # Start the timer here
    with util.ContextTimer() as t:
        dcksd = mct.DC_KSD(p, q, k, k, seed=r+11, alpha=alpha,
                           varest=varest,)
        dcksd_result = dcksd.perform_test(datr)

    return {
        'test_result': dcksd_result,
        'time_secs': t.secs,
    }


def met_imqksd_cov(P, Q, data_source, n, r):
    """
    KSD-based model comparison test
        * One covariance-preconditioned IMQ kernel for the two statistics.
        * Requires exact marginals of the two models.
        * Use jackknife U-statistic variance estimator
    """
    kernel_data = sample_pqr(None, None, data_source, n=kernel_datasize, r=0, only_from_r=True)
    X = kernel_data.data()
    cov = np.cov(X, rowvar=False)
    k = kernel.KPIMQ(P=cov)

    dcksd_result = _met_ksd(P, Q, data_source, n, r, k=k,)
    return dcksd_result


def met_imqksd_med(P, Q, data_source, n, r):
    """
    KSD-based model comparison test
        * One IMQ kernel for the two statistics.
        * Requires exact marginals of the two models.
        * Use U-statistic variance estimator
    """
    # sample some data 
    kernel_data = sample_pqr(None, None, data_source, n=kernel_datasize, r=0, only_from_r=True)

    X = kernel_data.data()
    d = X.shape[1]
    medX = util.meddistance(X)
    k = kernel.KPIMQ(medX**2 * np.eye(d))


    dcksd_result = _met_ksd(P, Q, data_source, n, r, k=k,)
    return dcksd_result


def met_imqksd_med_balltrunc(P, Q, data_source, n, r, 
                    varest=util.second_order_ustat_variance_jackknife,
):
    """
    KSD-based model comparison test
        * One Median-scaled IMQ kernel for the two statistics.
        * Use jackknife variance estimator
    """

    # sample some data 
    kernel_data = sample_pqr(None, None, data_source, n=kernel_datasize, r=0, only_from_r=True)
    X = kernel_data.data()
    d = X.shape[1]
    medX = util.meddistance(X)
    kimq = kernel.KPIMQ(np.eye(d) * medX**2)
    r = P.radius
    w = kernel.WeightSmoothBall(radius=r, frac=0.9)
    kw = kernel.KSTWeight(w)
    k = kernel.KSTProduct(kimq, kw)
    dz = P.dim_l

    dcksd_result = _met_ksd(P, Q, data_source, n, r, k=k,)
    return dcksd_result



def met_gksd_med(P, Q, data_source, n, r,
                 varest=util.second_order_ustat_variance_jackknife,
                 ):
    """
    KSD-based model comparison test
        * One Gaussian kernel for the two statistics.
        * Requires exact marginals of the two models.
        * Use jackknife U-statistic variance estimator
    """
    # sample some data 
    kernel_data = sample_pqr(None, None, data_source, n=kernel_datasize, r=0, only_from_r=True)

    medz = util.meddistance(kernel_data.data(), subsample=1000)
    k = kernel.KGauss(sigma2=medz**2)

    dcksd_result = _met_ksd(P, Q, data_source, n, r, k=k)
    return dcksd_result


def _met_lksd(P, Q, data_source, n, r, k, ps_p=None, ps_q=None,
              varest=util.second_order_ustat_variance_jackknife):
    """
    Wrapper for LKSD model comparison test (relative test). 
    Different tests can be defined depending on the input kernel k.
    """
    # sample some data 
    datr = sample_pqr(None, None, data_source, n, r, only_from_r=True)

    # Start the timer here
    with util.ContextTimer() as t:
        n_burnin_p = burnin_sizes.get(type(P), 500)
        n_burnin_q = burnin_sizes.get(type(Q), 500)
        mc_param_p = MCParam(n_mcsamples, n_burnin_p)
        mc_param_q = MCParam(n_mcsamples, n_burnin_q)
        ldcksd = mct.LDC_KSD(P, Q, k, k, seed=r+11, alpha=alpha,
                             mc_param_p=mc_param_p, mc_param_q=mc_param_q,
                             ps_p=ps_p, ps_q=ps_q,
                             varest=varest,
                             )
        ldcksd_result = ldcksd.perform_test(datr)

    return {
            'test_result': ldcksd_result, 'time_secs': t.secs,
            }


def met_glksd_med(P, Q, data_source, n, r):
    """
    LKSD model comparison test
        * One Gaussian kernel for the two statistics.
        * Use jackknife variance estimator
    """
    # sample some data 
    kernel_data = sample_pqr(None, None, data_source, n=kernel_datasize, r=0, only_from_r=True)
    medz = util.meddistance(kernel_data.data(), subsample=1000)
    k = kernel.KGauss(sigma2=medz**2)

    ldcksd_result =  _met_lksd(P, Q, data_source, n, r, k=k, )
    return ldcksd_result
    

def met_imqlksd_med(P, Q, data_source, n, r, 
                    varest=util.second_order_ustat_variance_jackknife,
):
    """
    KSD-based model comparison test
        * One Median-scaled IMQ kernel for the two statistics.
        * Use jackknife variance estimator
    """

    # sample some data 
    kernel_data = sample_pqr(None, None, data_source, n=kernel_datasize, r=0, only_from_r=True)
    X = kernel_data.data()
    d = X.shape[1]
    medX = util.meddistance(X)
    k = kernel.KPIMQ(medX**2 * np.eye(d))

    ldcksd_result = _met_lksd(P, Q, data_source, n, r, k=k, varest=varest)
    return ldcksd_result


def met_imqlksd_med_ustatvar(P, Q, data_source, n, r, 
                    varest=util.second_order_ustat_variance_ustat,
):
    """
    KSD-based model comparison test
        * One Median-scaled IMQ kernel for the two statistics.
        * Use U-stat variance estimator
    """
    varest = util.second_order_ustat_variance_ustat
    return met_imqlksd_med(P, Q, data_source, n, r, varest=varest)


def met_imqlksd_med_vstatvar(P, Q, data_source, n, r, 
                    varest=util.second_order_ustat_variance_ustat,
):
    """
    KSD-based model comparison test
        * One Median-scaled IMQ kernel for the two statistics.
        * Use V-stat variance estimator
    """
    varest = util.second_order_ustat_variance_vstat
    return met_imqlksd_med(P, Q, data_source, n, r, varest=varest)


def met_imqlksd_medtrunc(P, Q, data_source, n, r, 
                    varest=util.second_order_ustat_variance_jackknife,
):
    """
    KSD-based model comparison test
        * One Median-scaled IMQ kernel for the two statistics.
        * Use jackknife variance estimator
    """

    # sample some data 
    kernel_data = sample_pqr(None, None, data_source, n=kernel_datasize, r=0, only_from_r=True)
    X = kernel_data.data()
    d = X.shape[1]
    medX = util.meddistance(X)
    kimq = kernel.KPIMQ(np.eye(d) * medX**2)
    # kimq = kernel.KPIMQ(np.cov(X, rowvar=False))
    # k = kernel.KGauss(sigma2=medX**2)
    L = P.lim_lower * np.ones(d)
    U = P.lim_upper * np.ones(d)
    w = kernel.WeightSmoothIntevalProduct(L=L, U=U)
    kw = kernel.KSTWeight(w)
    k = kernel.KSTProduct(kimq, kw)
    dz = P.dim_l

    stepsize = dz**(-1./3) * 1e-3
    def ps_p(X, n_sample, seed, n_burnin=1):
        with util.NumpySeedContext(seed+133):
            Z_init = np.random.randn(n, dz)
        return mcmc.truncated_ppca_exchange_mala(X, n_sample, Z_init, model=P,
                              stepsize=stepsize, seed=seed, n_burnin=n_burnin)
        
    def ps_q(X, n_sample, seed, n_burnin=1):
        with util.NumpySeedContext(seed+134):
            Z_init = np.random.randn(n, dz)
        return mcmc.truncated_ppca_exchange_mala(X, n_sample, Z_init, model=Q,
                              stepsize=stepsize, seed=seed, n_burnin=n_burnin)

    ldcksd_result = _met_lksd(P, Q, data_source, n, r, k=k, ps_p=ps_p, ps_q=ps_q, varest=varest)
    return ldcksd_result


def met_imqlksd_med_balltrunc(P, Q, data_source, n, r, 
                    varest=util.second_order_ustat_variance_jackknife,
):
    """
    KSD-based model comparison test
        * One Median-scaled IMQ kernel for the two statistics.
        * Use jackknife variance estimator
    """

    # sample some data 
    kernel_data = sample_pqr(None, None, data_source, n=kernel_datasize, r=0, only_from_r=True)
    X = kernel_data.data()
    d = X.shape[1]
    medX = util.meddistance(X)
    kimq = kernel.KPIMQ(np.eye(d) * medX**2)
    # kimq = kernel.KPIMQ(np.cov(X, rowvar=False))
    #  k = kimq
    r = P.radius
    w = kernel.WeightSmoothBall(radius=r, frac=0.8)
    kw = kernel.KSTWeight(w)
    k = kernel.KSTProduct(kimq, kw)
    dz = P.dim_l

    stepsize = dz**(-1./3) * 1e-3
    def ps_p(X, n_sample, seed, n_burnin=1):
        with util.NumpySeedContext(seed+133):
            Z_init = np.random.randn(n, dz)
        return mcmc.truncated_ppca_exchange_mala(X, n_sample, Z_init, model=P,
                                                 stepsize=stepsize, seed=seed, 
                                                 n_burnin=n_burnin)
        
    def ps_q(X, n_sample, seed, n_burnin=1):
        with util.NumpySeedContext(seed+134):
            Z_init = np.random.randn(n, dz)
        return mcmc.truncated_ppca_exchange_mala(X, n_sample, Z_init, model=Q,
                                                 stepsize=stepsize, seed=seed, 
                                                 n_burnin=n_burnin)

    ldcksd_result = _met_lksd(P, Q, data_source, n, r, k=k, ps_p=ps_p, ps_q=ps_q, varest=varest)
    return ldcksd_result


def met_imqlksd_cov(P, Q, data_source, n, r):
    """
    KSD-based model comparison test
        * One preconditioned IMQ kernel for the two statistics.
        * Use jackknife variance estimator
    """

    # sample some data 
    kernel_data = sample_pqr(None, None, data_source, n=kernel_datasize, r=0, only_from_r=True)

    X = kernel_data.data()
    cov = np.cov(X, rowvar=False)
    k = kernel.KPIMQ(P=cov)

    ldcksd_result = _met_lksd(P, Q, data_source, n, r, k=k,)

    return ldcksd_result


# Define our custom Job, which inherits from base class IndependentJob
class Ex1Job(IndependentJob):
   
    def __init__(self, aggregator, P, Q, data_source, prob_label, rep, met_func, n):
        walltime = 60*59*24 
        #walltime = 60*59
        memory = 54272 # int(n*1e-2) + 50

        IndependentJob.__init__(self, aggregator, walltime=walltime,
                               memory=memory)
        # P, P are lkgof.model.LatentVariableModel
        self.P = P
        self.Q = Q
        self.data_source = data_source
        self.prob_label = prob_label
        self.rep = rep
        self.met_func = met_func
        self.n = n

    # we need to define the abstract compute method. It has to return an instance
    # of JobResult base class
    def compute(self):

        P = self.P
        Q = self.Q
        data_source = self.data_source 
        r = self.rep
        n = self.n
        met_func = self.met_func
        prob_label = self.prob_label

        logger.info("computing. %s. prob=%s, r=%d,\
                n=%d"%(met_func.__name__, prob_label, r, n))
        with util.ContextTimer() as t:
            job_result = met_func(P, Q, data_source, n, r)

            # create ScalarResult instance
            result = SingleResult(job_result)
            # submit the result to my own aggregator
            self.aggregator.submit_result(result)
            func_name = met_func.__name__

        logger.info("done. ex1: %s, prob=%s, r=%d, n=%d. Took: %.3g s "%(func_name,
            prob_label, r, n, t.secs))

        # save result
        fname = '%s-%s-n%d_r%d_a%.3f.p' \
                %(prob_label, func_name, n, r, alpha )
        glo.ex_save_result(ex, job_result, prob_label, fname)

# This import is needed so that pickle knows about the class Ex1Job.
# pickle is used when collecting the results from the submitted jobs.
from lkgof.ex.ex1_vary_n import Ex1Job
from lkgof.ex.ex1_vary_n import _met_mmd
from lkgof.ex.ex1_vary_n import _met_ksd
from lkgof.ex.ex1_vary_n import _met_lksd
from lkgof.ex.ex1_vary_n import met_gmmd_med
from lkgof.ex.ex1_vary_n import met_imqmmd_med
from lkgof.ex.ex1_vary_n import met_imqmmd_medtrunc
from lkgof.ex.ex1_vary_n import met_imqmmd_cov
from lkgof.ex.ex1_vary_n import met_gksd_med
from lkgof.ex.ex1_vary_n import met_imqksd_med
from lkgof.ex.ex1_vary_n import met_imqlksd_medtrunc
from lkgof.ex.ex1_vary_n import met_imqksd_med_balltrunc
from lkgof.ex.ex1_vary_n import met_imqksd_cov
from lkgof.ex.ex1_vary_n import met_glksd_med
from lkgof.ex.ex1_vary_n import met_imqlksd_med
from lkgof.ex.ex1_vary_n import met_imqlksd_medtrunc
from lkgof.ex.ex1_vary_n import met_imqlksd_med_balltrunc
from lkgof.ex.ex1_vary_n import met_imqlksd_cov
from lkgof.ex.ex1_vary_n import met_imqlksd_med_ustatvar
from lkgof.ex.ex1_vary_n import met_imqlksd_med_vstatvar

#--- experimental setting -----
ex = 1

# significance level of the test
alpha = 0.05

# repetitions for each sample size 
reps = 300

# kernel data size
kernel_datasize = 1000

# burnin size
burnin_sizes = {
    model.PPCA: 200,
    model.BoundedPPCA: 500,
    model.DPMIsoGaussBase: 1000,
}

# Markov chain sample size 
# n_mcsamples = 500
n_mcsamples = 2000 # for bppca

# tests to try
method_funcs = [ 
    # met_gmmd_med,
    # met_gksd_med,
    # met_glksd_med,
    met_imqmmd_med,
    met_imqksd_med,
    # met_imqlksd_med,
    # met_imqlksd_med_ustatvar,
    # met_imqlksd_med_vstatvar,
    met_imqmmd_cov,
    met_imqksd_cov,
    # met_imqlksd_cov,
    # met_imqmmd_medtrunc,
    # met_imqlksd_medtrunc,
    # met_imqlksd_med_balltrunc,
    # met_imqksd_med_balltrunc,
   ]

# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting already exists.
is_rerun = False
# ---------------------------


def make_ppca_prob(dx=10, dz=5, var=1.0,
                   ptbp=1., ptbq=1.,
                   seed=13, same=False, 
                   truncate=False,
                   bppca_bound=10.,
                   ):
    """LDA problem. Perturb one element of 
    the weight matrix.

    Args:
        dx (int, optional): Input dimension. Defaults to 10.
        dz (int, optional): Latent dimension. Defaults to 5.
        var (float, optional): Likelihood variance. Defaults to 1.0.
        ptbp (float, optional): Perturbation for model P. Defaults to 1..
        ptbq (float, optional): Perturbation for model Q. Defaults to 1..
        seed (int, optional): Random seed. Defaults to 13.
        same (bool, optional): Use identical models. Defaults to False.

    Returns:
        tuple: (model.PPCA, model.PPCA, data.DataSource)
        a tuple of (P, Q, R)
    """

    with util.NumpySeedContext(seed):
        weight = np.random.uniform(0, 1., [dx, dz])
    ppca_r = (model.PPCA(weight, var) if not truncate
              else model.BoundedPPCA(weight, var, lim_upper=bppca_bound, lim_lower=-bppca_bound))
    weightp = weight.copy()
    weightp[0, 0] = weightp[0, 0] + ptbp
    modelp = (model.PPCA(weightp, var) if not truncate 
              else model.BoundedPPCA(weightp, var, lim_upper=bppca_bound, lim_lower=-bppca_bound))
    if same:
        modelq = (model.PPCA(weightp, var) if not truncate 
              else model.BoundedPPCA(weightp, var, lim_upper=bppca_bound, lim_lower=-bppca_bound))
    else:
        weightq = weight.copy()
        weightq[0, 0] = weightq[0, 0] + ptbq
        modelq = (model.PPCA(weightq, var) if not truncate
                  else model.BoundedPPCA(weightq, var, lim_upper=bppca_bound, lim_lower=-bppca_bound))
    ds = ppca_r.get_datasource()

    return modelp, modelq, ds

def make_ballppca_prob(dx=10, dz=5, var=1.0,
                       ptbp=1., ptbq=1.,
                       seed=13, same=False,
                       radius=30.,
                   ):
    """LDA problem. Perturb one element of 
    the weight matrix.

    Args:
        dx (int, optional): Input dimension. Defaults to 10.
        dz (int, optional): Latent dimension. Defaults to 5.
        var (float, optional): Likelihood variance. Defaults to 1.0.
        ptbp (float, optional): Perturbation for model P. Defaults to 1..
        ptbq (float, optional): Perturbation for model Q. Defaults to 1..
        seed (int, optional): Random seed. Defaults to 13.
        same (bool, optional): Use identical models. Defaults to False.
        radius (float, optional): Radius of the domain. Defaults to 30.

    Returns:
        tuple: (model.BallPPCA, model.BallPPCA, data.DataSource)
        a tuple of (P, Q, R)
    """

    with util.NumpySeedContext(seed):
        weight = np.random.uniform(0, 1., [dx, dz])
    ppca_r = model.BallPPCA(weight, var, radius=radius)
    weightp = weight.copy()
    weightp[0, 0] = weightp[0, 0] + ptbp
    modelp = model.BallPPCA(weightp, var, radius=radius) 
    if same:
        modelq = model.BallPPCA(weightp, var, radius=radius)
    else:
        weightq = weight.copy()
        weightq[0, 0] = weightq[0, 0] + ptbq
        modelq = model.BallPPCA(weightq, var, radius=radius)
    ds = ppca_r.get_datasource()

    return modelp, modelq, ds


def make_dpm_isogauss_prob(tr_size=10, dx=10, var=2., prior_var=1., ptbp=1., ptbq=2.,
                           seed=13):
    """Gaussian Dirichlet process mixture problem.
    Perturb the prior mean.

    Args:
        tr_size (int, optional): Training dataset size. Defaults to 10.
        dx (int, optional): Input dimension. Defaults to 10.
        var (float, optional): Variance of likelihood. Defaults to 2..
        prior_var (float, optional): Prior's variance. Defaults to 1..
        ptbp (float, optional): Perturbation for P. Defaults to 1..
        ptbq (float, optional): Perturbation for Q. Defaults to 2..
        seed (int, optional): Random seed. Defaults to 13.

    Returns:
        tuple: (model.DPMIsoGaussBase, model.DPMIsoGaussBase, data.DataSource)
        a tuple of (P, Q, R)

    """
    mean = np.zeros([dx]) # same as prior mean
    var = var + prior_var
    ds = data.DSIsotropicNormal(mean, var)
    obs = ds.sample(tr_size, seed=seed).data()
    
    ptb_dir = np.ones([dx]) / dx**0.5
    modelp = model.DPMIsoGaussBase(var, mean+ptb_dir*ptbp, prior_var, obs=obs)
    modelq = model.DPMIsoGaussBase(var, mean+ptb_dir*ptbq, prior_var, obs=obs)
    return modelp, modelq, ds


def get_ns_pqrsource(prob_label):
    """
    Return (ns, P, Q, ds), a tuple of
    - ns: a list of sample sizes n's
    - P: a lkgof.model.LatentVariableModel representing the model P
    - Q: a lkgof.model.LatentVariableModel representing the model Q
    - ds: a DataSource. The DataSource generates sample from R.

    * (P, Q, ds) together specity a three-sample (or model comparison) problem.
    """

    prob2tuples = { 
        'ppca_h0_dx10_dz5':
            # list of sample sizes
            ([100, 200, 300], ) + make_ppca_prob(dx=10, dz=5, ptbp=1., ptbq=2.),
        'ppca_h1_dx10_dz5':
            # list of sample sizes
            ([100, 200, 300, 400, 500], ) + make_ppca_prob(dx=10, dz=5, ptbp=2., ptbq=1.,),
        'ppca_h0_dx50_dz10':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ppca_prob(dx=50, dz=10, ptbp=1., ptbq=2.),
        'ppca_h0_dx50_dz10_p0_q1':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ppca_prob(dx=50, dz=10, ptbp=0., ptbq=1.),
        'ppca_h0_dx50_dz10_p1_q1':
            # list of sample sizes
            ([i*100 for i in range(1, 3+1)], ) + make_ppca_prob(dx=50, dz=10, ptbp=1., ptbq=1.),
        'ppca_h0_dx100_dz10_p1_q1':
            # list of sample sizes
            ([i*100 for i in range(1, 3+1)], ) + make_ppca_prob(dx=100, dz=10, ptbp=1., ptbq=1.),
        'ppca_h0_dx50_dz10_p1_q1+1e-4':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ppca_prob(dx=50, dz=10, ptbp=1., ptbq=1.+1e-4),
        'ppca_h0_dx50_dz10_p1_q1+1e-10':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ppca_prob(dx=50, dz=10, ptbp=1., ptbq=1.+1e-10),
        'ppca_h1_dx100_dz10':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ppca_prob(dx=100, dz=10, ptbp=2.),
        'ppca_h1_dx100_dz10_p3_q1':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ppca_prob(dx=100, dz=10, ptbp=3.),
        'ppca_h0_dx100_dz10_p1_q1+1e-5':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ppca_prob(dx=100, dz=10, ptbp=1., ptbq=1.+1e-5),
        'bppca_h0_dx5_dz2_p1_q1+1e-5':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ppca_prob(dx=5, dz=2, ptbp=1., 
                                                                ptbq=1.+1e-5, truncate=True),
        'bppca_h1_dx5_dz2_p1+1e-1_q1':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ppca_prob(dx=5, dz=2, ptbp=1.+1e-1,
                                                                truncate=True),
        'ballppca_h1_dx100_dz10_p2_q1_r50':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ballppca_prob(dx=100, dz=10, ptbp=2, ptbq=1,
                                                                    radius=50),
        'ballppca_h1_dx100_dz10_p2_q1_r40':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ballppca_prob(dx=100, dz=10, ptbp=2, ptbq=1,
                                                                    radius=40),
        'ballppca_h1_dx100_dz10_p3_q1_r50':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ballppca_prob(dx=100, dz=10, ptbp=3, ptbq=1,
                                                                    radius=50),
        'ballppca_h0_dx100_dz10_p1_q1+1e-5_r50':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ballppca_prob(dx=100, dz=10, ptbq=1+1e-5,
                                                                    radius=50),
        'bppca_h1_dx100_dz10_p2_q1_b10':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ppca_prob(dx=100, dz=10, ptbp=2,
                                                                truncate=True, bppca_bound=10),
        'bppca_h1_dx100_dz10_p3_q1_b5':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ppca_prob(dx=100, dz=10, ptbp=3,
                                                                truncate=True, bppca_bound=5),
        'bppca_h1_dx100_dz10_p2_q1_b7':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ppca_prob(dx=100, dz=10, ptbp=2,
                                                                truncate=True, bppca_bound=7),
        'bppca_h0_dx100_dz10_p1_q1+1e-5_b10':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ppca_prob(dx=100, dz=10, ptbq=1.+1e-5,
                                                                truncate=True, bppca_bound=10),
        'bppca_h0_dx100_dz10_p1_q1+1e-5_b7':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ppca_prob(dx=100, dz=10, ptbq=1.+1e-5,
                                                                truncate=True, bppca_bound=7),
        'bppca_h0_dx100_dz10_p1_q1+1e-5_b5':
            # list of sample sizes
            ([i*100 for i in range(1, 5+1)], ) + make_ppca_prob(dx=100, dz=10, ptbq=1.+1e-5,
                                                                truncate=True, bppca_bound=5),
        'isogdpm_h0_dx10_tr10_p1_q2':
            # list of sample sizes
            ([i*100 for i in range(1, 3+1)], ) + make_dpm_isogauss_prob(tr_size=10, dx=50, ptbp=1., ptbq=2),
        'isogdpm_h0_dx10_tr50_p1_q5':
            # list of sample sizes
            ([i*100 for i in range(1, 3+1)], ) + make_dpm_isogauss_prob(tr_size=50, dx=10, ptbp=1., ptbq=5),
        'isogdpm_h1_dx10_tr10_p2_q1':
            # list of sample sizes
            ([i*100 for i in range(1, 3+1)], ) + make_dpm_isogauss_prob(tr_size=10, dx=50, ptbp=1., ptbq=2),
        'isogdpm_h1_dx10_tr50_p5_q1':
            # list of sample sizes
            ([i*100 for i in range(1, 3+1)], ) + make_dpm_isogauss_prob(tr_size=50, dx=10, ptbp=5., ptbq=1),

    }  # end of prob2tuples
    if prob_label not in prob2tuples:
        raise ValueError('Unknown problem label. Need to be one of %s'%str(list(prob2tuples.keys()) ))
    return prob2tuples[prob_label]


def run_problem(prob_label):
    """Run the experiment"""
    # ///////  submit jobs //////////
    # create folder name string
    #result_folder = glo.result_folder()
    from lkgof.config import expr_configs
    tmp_dir = expr_configs['scratch_path']
    foldername = os.path.join(tmp_dir, 'lkmod_slurm', 'e%d'%ex)
    logger.info("Setting engine folder to %s" % foldername)

    # create parameter instance that is needed for any batch computation engine
    logger.info("Creating batch parameter instance")
    batch_parameters = BatchClusterParameters(
        foldername=foldername, job_name_base="e%d_"%ex, parameter_prefix="")

    # Use the following line if Slurm queue is not used.
    # engine = SerialComputationEngine()
    partitions = expr_configs['slurm_partitions']
    if partitions is None:
        engine = SlurmComputationEngine(batch_parameters)
    else:
        engine = SlurmComputationEngine(batch_parameters, partition=partitions)
    n_methods = len(method_funcs)

    # problem setting
    ns, P, Q, ds, = get_ns_pqrsource(prob_label)

    # repetitions x len(ns) x #methods
    aggregators = np.empty((reps, len(ns), n_methods ), dtype=object)

    for r in range(reps):
        for ni, n in enumerate(ns):
            for mi, f in enumerate(method_funcs):
                # name used to save the result
                func_name = f.__name__
                fname = '%s-%s-n%d_r%d_a%.3f.p' \
                        %(prob_label, func_name, n, r, alpha,)
                if not is_rerun and glo.ex_file_exists(ex, prob_label, fname):
                    logger.info('%s exists. Load and return.'%fname)
                    job_result = glo.ex_load_result(ex, prob_label, fname)

                    sra = SingleResultAggregator()
                    sra.submit_result(SingleResult(job_result))
                    aggregators[r, ni, mi] = sra
                else:
                    # result not exists or rerun
                    job = Ex1Job(SingleResultAggregator(), P, Q, ds, prob_label,
                            r, f, n)

                    agg = engine.submit_job(job)
                    aggregators[r, ni, mi] = agg

    # let the engine finish its business
    logger.info("Wait for all call in engine")
    engine.wait_for_all()

    # ////// collect the results ///////////
    logger.info("Collecting results")
    job_results = np.empty((reps, len(ns), n_methods), dtype=object)
    for r in range(reps):
        for ni, n in enumerate(ns):
            for mi, f in enumerate(method_funcs):
                logger.info("Collecting result (%s, r=%d, n=%d)" %
                        (f.__name__, r, n))
                # let the aggregator finalize things
                aggregators[r, ni, mi].finalize()

                # aggregators[i].get_final_result() returns a SingleResult instance,
                # which we need to extract the actual result
                job_result = aggregators[r, ni, mi].get_final_result().result
                job_results[r, ni, mi] = job_result

    #func_names = [f.__name__ for f in method_funcs]
    #func2labels = exglobal.get_func2label_map()
    #method_labels = [func2labels[f] for f in func_names if f in func2labels]

    # save results 
    results = {'job_results': job_results, 
            'P': P, 'Q': Q,
            'data_source': ds, 
            'alpha': alpha, 'repeats': reps, 'ns': ns,
            'method_funcs': method_funcs, 'prob_label': prob_label,
            }
    
    # class name 
    fname = 'ex%d-%s-me%d_rs%d_nmi%d_nma%d_a%.3f.p' \
        %(ex, prob_label, n_methods, reps, min(ns), max(ns), alpha,)

    glo.ex_save_result(ex, results, fname)
    logger.info('Saved aggregated results to %s'%fname)

#---------------------------
def main():
    if len(sys.argv) != 2:
        print('Usage: %s problem_label'%sys.argv[0])
        sys.exit(1)
    prob_label = sys.argv[1]
    run_problem(prob_label)

if __name__ == '__main__':
    main()

