"""Simulation to get the test power vs increasing sample size. Discrete observations."""

import os 
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import lkgof.util as util
from lkgof import model
from lkgof import mctest as mct
from kmod.mctest import SC_MMD
import lkgof.glo as glo
import lkgof.kernel as kernel
import scipy.stats as stats
from lkgof.goftest import MCParam


# need independent_jobs package 
# https://github.com/wittawatj/independent-jobs
# The independent_jobs has to be in the global search path (.bashrc)
import independent_jobs as inj
from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.SingleResult import SingleResult
from independent_jobs.aggregators.SingleResultAggregator import SingleResultAggregator
from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.tools.Log import logger

import autograd.numpy as np
import os
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
    - P: a lkgof.model.LatentVariableModel (candidate model 1)
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


def sample_pqr(ds_p, ds_q, ds_r, n, r, only_from_r=False):
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
    datp = ds_p.sample(n, seed=r+10000)
    datq = ds_q.sample(n, seed=r+20000)
    return datp, datq, datr


# -------------------------------------------------------
def met_dis_hmmd(P, Q, data_source, n, r):
    """
    Bounliphone et al., 2016's MMD-based 3-sample test.
    * Exponentiated Hamming distance kernel for discrete observations.
    * Use full sample for testing (no holding out for optimization)
    """
    if not P.has_datasource() or not Q.has_datasource():
        # Not applicable. Return {}.
        return {}

    ds_p = P.get_datasource()
    ds_q = Q.get_datasource()
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim

    # sample some data 
    datp, datq, datr = sample_pqr(ds_p, ds_q, data_source, n, r, only_from_r=False)

    # Start the timer here
    with util.ContextTimer() as t:
        # X, Y, Z = datp.data(), datq.data(), datr.data()
        k = kernel.KHamming(n_values, d)
        scmmd = SC_MMD(datp, datq, k, alpha=alpha)
        scmmd_result = scmmd.perform_test(datr)

    return {
            'test_result': scmmd_result, 'time_secs': t.secs}


def _met_dis_mmd(P, Q, data_source, n, r, k):
    """
    Wrapper to define Bounliphone et al., 2016's MMD-based 3-sample test.
    Different test are defined depending on the input kernel k.
    * Use full sample for testing (no holding out for optimization)
    """
    if not P.has_datasource() or not Q.has_datasource():
        # Not applicable. Return {}.
        return {}

    ds_p = P.get_datasource()
    ds_q = Q.get_datasource()
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))

    # sample some data 
    datp, datq, datr = sample_pqr(ds_p, ds_q, data_source, n, r, only_from_r=False)

    # Start the timer here
    with util.ContextTimer() as t:
        # X, Y, Z = datp.data(), datq.data(), datr.data()
        scmmd = SC_MMD(datp, datq, k, alpha=alpha)
        scmmd_result = scmmd.perform_test(datr)

    return {
            'test_result': scmmd_result, 'time_secs': t.secs}


def met_dis_bowmmd(P, Q, data_source, n, r):
    """
    Bounliphone et al., 2016's MMD-based 3-sample test.
    * Bag of words kernel for discrete observations.
    * Use full sample for testing (no holding out for optimization)
    """
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim
    k = kernel.KBoW(n_values, d)
    result = _met_dis_mmd(P, Q, data_source, n, r, k)
    return result


def met_dis_nbowmmd(P, Q, data_source, n, r):
    """
    Bounliphone et al., 2016's MMD-based 3-sample test.
    * Normalized BoW kernel for discrete observations.
    * Use full sample for testing (no holding out for optimization)
    """
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim
    k = kernel.KNormalizedBoW(n_values, d)
    result = _met_dis_mmd(P, Q, data_source, n, r, k)
    return result


def met_dis_gbowmmd(P, Q, data_source, n, r):
    """
    Bounliphone et al., 2016's MMD-based 3-sample test.
    * Gaussian BoW kernel for discrete observations.
    * Use full sample for testing (no holding out for optimization)
    """
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim
    k = kernel.KGaussBoW(n_values, d)
    result = _met_dis_mmd(P, Q, data_source, n, r, k)
    return result


def met_dis_hksd(P, Q, data_source, n, r):
    """
    KSD-based model comparison test (relative test). 
        * Exponentiated Hamming distance kernel for discrete observations.
        * Requires exact marginals of the two models.
    """
    if not P.has_unnormalized_density() or not Q.has_unnormalized_density():
        # Not applicable. Return {}.
        return {}

    p = P.get_unnormalized_density()
    q = Q.get_unnormalized_density()
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim

    # sample some data 
    datr = sample_pqr(None, None, data_source, n, r, only_from_r=True)

    # Start the timer here
    with util.ContextTimer() as t:
        k = kernel.KHamming(n_values, d)

        dcksd= mct.DC_KSD(p, q, k, k, seed=r+11, alpha=alpha)
        dcksd_result = dcksd.perform_test(datr)

    return {
            'test_result': dcksd_result, 'time_secs': t.secs}


def _met_dis_lksd(P, Q, data_source, n, r, k, mc_sample=500,
                  varest=util.second_order_ustat_variance_ustat,
                  ):
    """
    Wrapper for Latent KSD-based model comparison test (relative test). 
    Different test are defined depending on the input kernel k.
    """
    # sample some data 
    datr = sample_pqr(None, None, data_source, n, r, only_from_r=True)

    # Start the timer here
    with util.ContextTimer() as t:
        n_burnin_p = burnin_sizes.get(type(P), 500)
        n_burnin_q = burnin_sizes.get(type(Q), 500)
        mc_param_p = MCParam(mc_sample, n_burnin_p)
        mc_param_q = MCParam(mc_sample, n_burnin_q)
        ldcksd = mct.LDC_KSD(P, Q, k, k, seed=r+11, alpha=alpha,
                             mc_param_p=mc_param_p, mc_param_q=mc_param_q,
                             varest=varest,
                             )
        ldcksd_result = ldcksd.perform_test(datr)

    return {
            'test_result': ldcksd_result, 'time_secs': t.secs,
            }


def met_dis_hlksd(P, Q, data_source, n, r):
    """
    Latent KSD-based model comparison test (relative test). 
        * Exponentiated Hamming distance kernel for discrete observations.
    """
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim

    k = kernel.KHamming(n_values, d)
    result = _met_dis_lksd(P, Q, data_source, n, r, k)
    return result


def met_dis_bowlksd(P, Q, data_source, n, r):
    """
    Latent KSD-based model comparison test (relative test). 
        * Exponentiated Vanilla BoW kernel for discrete observations.
        * Use U-statistic variance estimator
    """


    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim

    k = kernel.KBoW(n_values, d)
    result = _met_dis_lksd(P, Q, data_source, n, r, k)
    return result


def met_dis_bowlksd_vstat(P, Q, data_source, n, r):
    """
    Latent KSD-based model comparison test (relative test). 
        * Exponentiated Vanilla BoW kernel for discrete observations.
        * Use V-statistic variance estimator
    """

    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim
    k = kernel.KBoW(n_values, d)
    result = _met_dis_lksd(P, Q, data_source, n, r, k,
                           varest=util.second_order_ustat_variance_vstat,
                           )
    return result


def met_dis_nbowlksd(P, Q, data_source, n, r):
    """
    Latent KSD-based model comparison test (relative test). 
        * Exponentiated Normalized BoW kernel for discrete observations.
        * Use U-statistic variance estimator
    """
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim

    k = kernel.KNormalizedBoW(n_values, d)
    result = _met_dis_lksd(P, Q, data_source, n, r, k)
    return result


def met_dis_nbowlksd_vstat(P, Q, data_source, n, r):
    """
    Latent KSD-based model comparison test (relative test). 
        * Exponentiated Normalized BoW kernel for discrete observations.
        * Use V-statistic variance estimator
    """
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim

    k = kernel.KNormalizedBoW(n_values, d)
    result = _met_dis_lksd(P, Q, data_source, n, r, k,
                           varest=util.second_order_ustat_variance_vstat,
                           )
    return result


def met_dis_gbowlksd(P, Q, data_source, n, r):
    """
    Latent KSD-based model comparison test (relative test).
        * Exponentiated Gaussian BoW kernel for discrete observations.
        * Use U-statistic variance estimator
    """
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim

    k = kernel.KGaussBoW(n_values, d)
    result = _met_dis_lksd(P, Q, data_source, n, r, k, )
    return result


def met_dis_gbowlksd_vstat(P, Q, data_source, n, r):
    """
    Latent KSD-based model comparison test (relative test).
        * Exponentiated Gaussian BoW kernel for discrete observations.
        * Use V-statistic variance estimator
    """
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim

    k = kernel.KGaussBoW(n_values, d)
    result = _met_dis_lksd(P, Q, data_source, n, r, k,
                           varest=util.second_order_ustat_variance_vstat,
                           )
    return result


def met_dis_gbowlksd_jackknife(P, Q, data_source, n, r):
    """
    Latent KSD-based model comparison test (relative test).
        * Exponentiated Gaussian BoW kernel for discrete observations.
        * Use a jackknife variance estimator
    """
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim

    k = kernel.KGaussBoW(n_values, d)
    result = _met_dis_lksd(P, Q, data_source, n, r, k,
                           varest=util.second_order_ustat_variance_jackknife,
                           )
    return result


# Define our custom Job, which inherits from base class IndependentJob
class Ex2Job(IndependentJob):
   
    def __init__(self, aggregator, P, Q, data_source, prob_label, rep, met_func, n):
        walltime = 60*59*24 
        #walltime = 60*59
        memory = 52472#int(n*1e-2) + 50

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

        logger.info("done. ex2: %s, prob=%s, r=%d, n=%d. Took: %.3g s "%(func_name,
            prob_label, r, n, t.secs))

        # save result
        fname = '%s-%s-n%d_r%d_a%.3f.p' \
                %(prob_label, func_name, n, r, alpha )
        glo.ex_save_result(ex, job_result, prob_label, fname)

# This import is needed so that pickle knows about the class Ex2Job.
# pickle is used when collecting the results from the submitted jobs.
from lkgof.ex.ex2_vary_n_disc import Ex2Job
from lkgof.ex.ex2_vary_n_disc import _met_dis_mmd
from lkgof.ex.ex2_vary_n_disc import met_dis_bowmmd
from lkgof.ex.ex2_vary_n_disc import met_dis_nbowmmd
from lkgof.ex.ex2_vary_n_disc import met_dis_gbowmmd
from lkgof.ex.ex2_vary_n_disc import met_dis_hmmd
from lkgof.ex.ex2_vary_n_disc import met_dis_hksd
from lkgof.ex.ex2_vary_n_disc import _met_dis_lksd
from lkgof.ex.ex2_vary_n_disc import met_dis_hlksd
from lkgof.ex.ex2_vary_n_disc import met_dis_bowlksd
from lkgof.ex.ex2_vary_n_disc import met_dis_bowlksd_vstat
from lkgof.ex.ex2_vary_n_disc import met_dis_nbowlksd
from lkgof.ex.ex2_vary_n_disc import met_dis_nbowlksd_vstat
from lkgof.ex.ex2_vary_n_disc import met_dis_gbowlksd
from lkgof.ex.ex2_vary_n_disc import met_dis_gbowlksd_vstat
from lkgof.ex.ex2_vary_n_disc import met_dis_gbowlksd_jackknife

#--- experimental setting -----
ex = 2 

# significance level of the test
alpha = 0.05

# repetitions for each sample size 
reps = 300

burnin_sizes = {
    model.LDAEmBayes: 5000,
}
# tests to try
method_funcs = [ 
    # met_dis_hmmd
    # met_dis_bowmmd,
    # met_dis_nbowmmd,
    # met_dis_gbowmmd,
    # met_dis_hksd,
    # met_dis_hlksd,
    # met_dis_bowlksd,
    # met_dis_bowlksd_vstat,
    # met_dis_nbowlksd_vstat,
    met_dis_gbowlksd_vstat,
    met_dis_gbowlksd,
    met_dis_gbowlksd_jackknife,
   ]

# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting already exists.
is_rerun = False
#---------------------------


def make_betabinom_prob(dx=10, n_trial=5, 
                        aptbp=0., bptbp=0,
                        aptbq=0., bptbq=0,
                        seed=13, same=False):
    """Beta binomial problem. 

    Args:
        dx (int, optional): dimension. Defaults to 10.
        n_trial (int, optional): the number of trials. Defaults to 5.
        aptbp (float, optional):
            perturbation to alpha param in beta prior. Defaults to 0..
        bptbp (float, optional):
            perturbation to beta param in bete prior. Defaults to 0.
        aptbq (float, optional):
            perturbation to alpha param in beta prior. Defaults to 0..
        bptbq (float, optional):
            perturbation to beta param in beta prior. Defaults to 0.
        seed (int, optional): Random seed. Defaults to 13.
        same (bool, optional): Use identical models. Defaults to False.

    Returns:
        tuple: (P, Q, data source for R)
    """
    n_trials = np.ones(dx, np.int64) * n_trial
    with util.NumpySeedContext(seed):
        alpha = np.random.uniform(2, 3)
        beta = np.random.uniform(3, 4)
    modelr = model.BetaBinomSinglePrior(n_trials, alpha, beta)
    modelp = model.BetaBinomSinglePrior(
        n_trials, alpha+aptbp, beta+bptbp)
    modelq = model.BetaBinomSinglePrior(
        n_trials, alpha+aptbq, beta+bptbq)
    ds = modelr.get_datasource()
    return modelp, modelq, ds


def make_lda_prob(n_words, n_topics, vocab_size,
                  ptb_p, ptb_q, seed=149):
    """LDA problem. 
    Perturbation is applied to the sparsity parameter of
    the Dirichlet prior. 

    Args:
        n_words:
            The number of words in a document. Input dimension.
        n_topics:
            The number of topics.
        vocab_size:
            Vocabulary size. The size of the lattice domain.
        ptb_p:
            pertubation parameter for model P
        ptb_q:
            perturbation parameter for model Q

    Returns:
        tuple: (model.LDAEmBayes, model.LDAEmBayes, data source for R)
            tuple of (P, Q, R)
    """

    n_values = np.ones(n_words, dtype=np.int64) * vocab_size
    TEMP = 1.
    with util.NumpySeedContext(seed):
        beta = stats.dirichlet(alpha=TEMP*np.ones(vocab_size)).rvs(size=n_topics)
    alpha = 0.1*np.ones([n_topics])
    modelr = model.LDAEmBayes(alpha, beta, n_values)
    alpha_p = alpha.copy()
    alpha_q = alpha.copy()
    beta_p = beta.copy()
    beta_q = beta.copy()
    alpha_p += ptb_p
    alpha_q += ptb_q
    modelp = model.LDAEmBayes(alpha_p, beta_p, n_values)
    modelq = model.LDAEmBayes(alpha_q, beta_q, n_values)
    ds = modelr.get_datasource()
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
        # A case where H0 (P is better) is true.
        'betabinom_h0_dx10_n5_p1p11q21':
            # list of sample sizes
            ([100, 200, 300], ) + make_betabinom_prob(dx=10, aptbp=2,
                                                      bptbp=1, aptbq=1., bptbq=1.,),
        'betabinom_h1_dx10_n5_p1p21q11':
            # list of sample sizes
            ([100, 200, 300], ) + make_betabinom_prob(dx=10, aptbp=1,
                                                      bptbp=1, aptbq=2., bptbq=1.,),
        'betabinom_h1_dx10_n1_p1p51q11':
            # list of sample sizes
            ([100, 200, 300], ) + make_betabinom_prob(dx=10, n_trial=1, aptbp=5.,
                                                      bptbp=1, aptbq=1., bptbq=1.,),
        'betabinom_h0_dx10_n1_p1p11q51':
            # list of sample sizes
            ([100, 200, 300], ) + make_betabinom_prob(dx=10, n_trial=1, aptbp=1.,
                                                      bptbp=1, aptbq=5., bptbq=1.,),
        'lda_h1_dx100_v100_t3_p1q05':
            ([100, 200, 300,], ) + make_lda_prob(n_words=100, n_topics=3,
                                                  vocab_size=100,
                                                  ptb_p=1., ptb_q=5e-1),
        'lda_h0_dx100_v100_t3_p05q1':
            ([100, 200, 300, ], ) + make_lda_prob(n_words=100, n_topics=3,
                                                  vocab_size=100,
                                                  ptb_p=5e-1, ptb_q=1.),
        'lda_h0_dx100_v100_t3_p05q06':
            ([100, 200, 300, ], ) + make_lda_prob(n_words=100, n_topics=3,
                                                  vocab_size=100,
                                                  ptb_p=5e-1, ptb_q=0.6),
        'lda_h1_dx100_v5_t3_p1q05':
            ([100, 200, 300], ) + make_lda_prob(n_words=100, n_topics=3,
                                                vocab_size=5,
                                                ptb_p=1., ptb_q=0.5),
        'lda_h0_dx50_v100_t3_p05q05':
            ([100, 200, 300], ) + make_lda_prob(n_words=50, n_topics=3,
                                                vocab_size=100,
                                                ptb_p=0.5, ptb_q=0.5),
        'lda_h0_dx50_v100_t3_p1q1':
            ([100, 200, 300, 400], ) + make_lda_prob(n_words=50, n_topics=3,
                                                vocab_size=100,
                                                ptb_p=1., ptb_q=1.),
        'lda_h0_dx100_v100_t3_p01q05':
            ([100, 200, 300], ) + make_lda_prob(n_words=50, n_topics=3,
                                                vocab_size=100,
                                                ptb_p=0.1, ptb_q=0.5),
        'lda_h1_dx100_v100_t3_p1q05':
            ([100, 200, 300], ) + make_lda_prob(n_words=50, n_topics=3,
                                                vocab_size=100,
                                                ptb_p=1, ptb_q=0.5),
        'lda_h0_dx50_v100_t3_p1q1+1e-4':
            ([100, 200, 300, 400, 500], ) + make_lda_prob(n_words=50, n_topics=3,
                                                vocab_size=100,
                                                ptb_p=1, ptb_q=1+1e-4),
        'lda_h0_dx50_v100_t3_p1+1e-1q1':
            ([100, 200, 300, ], ) + make_lda_prob(n_words=50, n_topics=3,
                                                vocab_size=100,
                                                ptb_p=1.1, ptb_q=1),
        'lda_h0_dx50_v100_t3_p0q05':
            ([100, 200, 300], ) + make_lda_prob(n_words=50, n_topics=3,
                                                vocab_size=100,
                                                ptb_p=0., ptb_q=0.5),
        'lda_h0_dx50_v100_t3_p05q06':
            ([100, 200, 300], ) + make_lda_prob(n_words=50, n_topics=3,
                                                vocab_size=100,
                                                ptb_p=0.5, ptb_q=0.6),
        'lda_h0_dx50_v100_t3_p04q05':
            ([100, 200, 300], ) + make_lda_prob(n_words=50, n_topics=3,
                                                vocab_size=100,
                                                ptb_p=0.4, ptb_q=0.5),
        'lda_h0_dx50_v100_t3_p05q05+1e-2':
            ([100, 200, 300], ) + make_lda_prob(n_words=50, n_topics=3,
                                                vocab_size=100,
                                                ptb_p=0.5, ptb_q=0.51),
        'lda_h1_dx50_v100_t3_p1q05':
            ([100, 200, 300], ) + make_lda_prob(n_words=50, n_topics=3,
                                                vocab_size=100,
                                                ptb_p=1., ptb_q=0.5),
        'lda_h1_dx50_v100_t3_p1q08':
            ([100, 200, 300], ) + make_lda_prob(n_words=50, n_topics=3,
                                                vocab_size=100,
                                                ptb_p=1., ptb_q=0.8),
        'lda_h0_dx10_v5_t3_p05q1':
            ([100, 200, 300, 500, 800, 1100], ) + make_lda_prob(n_words=10, n_topics=3,
                                                vocab_size=5,
                                                ptb_p=5e-1, ptb_q=1.),
        'lda_h1_dx10_v5_t3_p1q05':
            ([100, 200, 300, 500, 800, 1100], ) + make_lda_prob(n_words=10, n_topics=3,
                                                vocab_size=5,
                                                ptb_p=1., ptb_q=5e-1),

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
                    job = Ex2Job(SingleResultAggregator(), P, Q, ds, prob_label,
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

