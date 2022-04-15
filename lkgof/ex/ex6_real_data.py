"""Simulation to get the test power vs increasing sample size. Discrete observations."""

from distutils.command.config import config
import os 
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import lkgof.util as util
import lkgof.data as data
from lkgof import model
from lkgof import mctest as mct
from lkgof.mctest import SC_MMD
from lkgof.config import expr_configs
import lkgof.glo as glo
import lkgof.kernel as kernel
import scipy.stats as stats
from lkgof.goftest import MCParam
from gensim.models import LdaModel
from gensim.test.utils import datapath


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

def sample_pqr(P, Q, rdata, n, r, only_from_r=False, 
               pqr_same_sample_size=False, mc_sample=None, 
               add_mcsample=True, add_burnin=True):
    """
    Generate three samples from the three data sources given a trial index r.
    All met_ functions should use this function to draw samples. This is to
    provide a uniform control of how samples are generated in each trial.

    ds_p: DataSource for model P
    ds_q: DataSource for model Q
    ds_r: Numpy array of the data to be subsampled. 
    n: sample size to draw from each
    r: trial index

    Return (datp, datq, datr) where each is a Data containing n x d numpy array
    Return datr if only_from_r is True.
    """
    dsize = rdata.shape[0]
    with util.NumpySeedContext(seed=r+30):
        idx = np.random.choice(dsize, n, replace=False)
        datr = data.Data(rdata[idx])
    if only_from_r:
        return datr
    ds_p = P.get_datasource()
    ds_q = Q.get_datasource()
    if pqr_same_sample_size:
        n_model_samples = n
    else:
        nmax = 10000
        if mc_sample is None:
            mc_sample = n_mcsamples
        mc_sample = mc_sample if add_mcsample else 0
        n_burnin_p = burnin_sizes.get(type(P), 500)
        n_burnin_q = burnin_sizes.get(type(Q), 500)
        n_burnin = max(n_burnin_p, n_burnin_q,) if add_burnin else 0
        n_model_samples = n + n_burnin + mc_sample
        n_model_samples = min(n_model_samples, nmax)
        logger.info('n_model_samples: {}'.format(n_model_samples))
    datp = ds_p.sample(n_model_samples, seed=r+1000)
    datq = ds_q.sample(n_model_samples, seed=r+2000)
    return datp, datq, datr


# -------------------------------------------------------


def _met_dis_mmd(P, Q, data_source, n, r, k, mc_sample=None, add_mcsample=True, add_burnin=True):
    """
    Wrapper to define Bounliphone et al., 2016's MMD-based 3-sample test.
    Different test are defined depending on the input kernel k.
    * Use full sample for testing (no holding out for optimization)
    """
    if not P.has_datasource() or not Q.has_datasource():
        # Not applicable. Return {}.
        return {}

    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))

        # Start the timer here
    with util.ContextTimer() as t:
        # sample some data 
        datp, datq, datr = sample_pqr(P, Q, data_source, n, r, only_from_r=False, 
                mc_sample=mc_sample, add_mcsample=add_mcsample, add_burnin=add_burnin)

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
    k = kernel.KGaussBoW(n_values, d, s2=1)
    result = _met_dis_mmd(P, Q, data_source, n, r, k)
    return result


def met_dis_imqbowmmd(P, Q, data_source, n, r):
    """
    Bounliphone et al., 2016's MMD-based 3-sample test.
    * IMQ BoW kernel for discrete observations.
    * Use full sample for testing (no holding out for optimization)
    """
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim
    k = kernel.KIMQBoW(n_values, d, s2=1)
    result = _met_dis_mmd(P, Q, data_source, n, r, k)
    return result

def met_dis_imqbowmmd_moremc(P, Q, data_source, n, r):
    """
    Bounliphone et al., 2016's MMD-based 3-sample test.
    * IMQ BoW kernel for discrete observations.
    * Use full sample for testing (no holding out for optimization)
    """
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim
    k = kernel.KIMQBoW(n_values, d, s2=1)
    result = _met_dis_mmd(P, Q, data_source, n, r, k, mc_sample=10000+n_mcsamples)
    return result

def met_dis_imqbowmmd_cheap(P, Q, data_source, n, r):
    """
    Bounliphone et al., 2016's MMD-based 3-sample test.
    * IMQ BoW kernel for discrete observations.
    * Use full sample for testing (no holding out for optimization)
    """
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim
    k = kernel.KIMQBoW(n_values, d, s2=1)
    result = _met_dis_mmd(P, Q, data_source, n, r, k, mc_sample=500, add_mcsample=True, add_burnin=False)
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


def _met_dis_lksd(P, Q, data_source, n, r, k, mc_sample=None,
                  varest=util.second_order_ustat_variance_jackknife,
                  ):
    """
    Wrapper for Latent KSD-based model comparison test (relative test). 
    Different test are defined depending on the input kernel k.
    """
    # sample some data 
    datr = sample_pqr(None, None, data_source, n, r, only_from_r=True)

    if mc_sample is None:
        mc_sample = n_mcsamples
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

    k = kernel.KGaussBoW(n_values, d, s2=1)
    result = _met_dis_lksd(P, Q, data_source, n, r, k, )
    return result


def met_dis_imqbowlksd(P, Q, data_source, n, r):
    """
    Latent KSD-based model comparison test (relative test).
        * IMQ BoW kernel for discrete observations.
        * Use U-statistic variance estimator
    """
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim

    k = kernel.KIMQBoW(n_values, d, s2=1)
    result = _met_dis_lksd(P, Q, data_source, n, r, k, )
    return result


def met_dis_imqbowlksd_moremc(P, Q, data_source, n, r):
    """
    Latent KSD-based model comparison test (relative test).
        * IMQ BoW kernel for discrete observations.
        * Use U-statistic variance estimator
        * Extra MC samples to compute the score
    """
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim

    k = kernel.KIMQBoW(n_values, d, s2=1)
    mc_sample = n_mcsamples + 10000
    result = _met_dis_lksd(P, Q, data_source, n, r, k, mc_sample=mc_sample)
    return result



# Define our custom Job, which inherits from base class IndependentJob
class Ex6Job(IndependentJob):
   
    def __init__(self, aggregator, prob_label, rep, met_func, n):
        walltime = 60*59*24 
        #walltime = 60*59
        memory = 52472#int(n*1e-2) + 50

        IndependentJob.__init__(self, aggregator, walltime=walltime,
                                memory=memory)
        # P, P are lkgof.model.LatentVariableModel
        # self.P = P
        # self.Q = Q
        # self.data_source = data_source
        self.prob_label = prob_label
        self.rep = rep
        self.met_func = met_func
        self.n = n

    # we need to define the abstract compute method. It has to return an instance
    # of JobResult base class
    def compute(self):

        # P = self.P
        # Q = self.Q
        # data_source = self.data_source 
        r = self.rep
        n = self.n
        met_func = self.met_func
        prob_label = self.prob_label

        logger.info("computing. %s. prob=%s, r=%d,\
                n=%d"%(met_func.__name__, prob_label, r, n))
        with util.ContextTimer() as t:
            job_result = met_func(P, Q, ds, n, r)
            # create ScalarResult instance
            result = SingleResult(job_result)
            # submit the result to my own aggregator
            self.aggregator.submit_result(result)
            func_name = met_func.__name__

        logger.info("done. ex6: %s, prob=%s, r=%d, n=%d. Took: %.3g s "%(func_name,
            prob_label, r, n, t.secs))

        # save result
        fname = '%s-%s-n%d_r%d_a%.3f.p' \
                %(prob_label, func_name, n, r, alpha )
        glo.ex_save_result(ex, job_result, prob_label, fname)

# This import is needed so that pickle knows about the class Ex6Job.
# pickle is used when collecting the results from the submitted jobs.
from lkgof.ex.ex6_real_data import Ex6Job
from lkgof.ex.ex6_real_data import _met_dis_mmd
from lkgof.ex.ex6_real_data import met_dis_gbowmmd
from lkgof.ex.ex6_real_data import _met_dis_lksd
from lkgof.ex.ex6_real_data import met_dis_gbowlksd
from lkgof.ex.ex6_real_data import met_dis_imqbowmmd
from lkgof.ex.ex6_real_data import met_dis_imqbowmmd_moremc
from lkgof.ex.ex6_real_data import met_dis_imqbowmmd_cheap
from lkgof.ex.ex6_real_data import met_dis_imqbowlksd
from lkgof.ex.ex6_real_data import met_dis_imqbowlksd_moremc

#--- experimental setting -----
ex = 6

# significance level of the test
alpha = 0.05

# repetitions for each sample size 
reps = 300

burnin_sizes = {
    model.LDAEmBayes: 500,
}

n_mcsamples = 5000

# tests to try
method_funcs = [ 
    # met_dis_bowmmd,
    # met_dis_nbowmmd,
    # met_dis_gbowmmd,
    met_dis_imqbowmmd,
    # met_dis_imqbowmmd_moremc,
    # met_dis_imqbowmmd_cheap,
    # met_dis_hksd,
    # met_dis_hlksd,
    # met_dis_bowlksd,
    # met_dis_gbowlksd,
    met_dis_imqbowlksd,
    #  met_dis_imqbowlksd_moremc,
   ]


# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting already exists.
is_rerun = False
#---------------------------

def make_lda_prob(problem, pname, qname, rname, seed=149):
    """LDA problem. 
    Perturbation is applied to the sparsity parameter of
    the Dirichlet prior. 

    Args:
        problem:
            Problem name, e.g., arxiv. String. 
        modelp:
            Dataset on which model P is trained on. String. 
        modelq:
            Dataset on which model P is trained on. String. 

    Returns:
        tuple: (model.LDAEmBayes, model.LDAEmBayes, data source for R)
            tuple of (P, Q, R)
    """
    datadir = os.path.join(expr_configs['problems_path'], 'arxiv')
    modeldir = os.path.join(datadir, 'models')
    samplefile = os.path.join(datadir, 'testdata', problem, 
            '{}.npy'.format(rname))
    datar = np.load(samplefile).astype(np.int64)
    n_words = datar.shape[1]

    categories = [pname, qname, rname]
    modelp_filename= 'LDA_{}'.format(pname)
    modelp_path = os.path.join(modeldir, problem, modelp_filename)
    lda = LdaModel.load(datapath(modelp_path))
    betap = lda.get_topics()
    vocab_size = betap.shape[1]
    n_values = vocab_size*np.ones(n_words, dtype=np.int64)
    alphap = np.load('{}.alpha.npy'.format(modelp_path))
    print(alphap)
    modelp = model.LDAEmBayes(alpha=alphap, beta=betap,
                              n_values=n_values)
    
    modelq_filename= 'LDA_{}'.format(qname)
    modelq_path = os.path.join(modeldir, problem, modelq_filename)
    lda = LdaModel.load(datapath(modelq_path))
    alphaq = np.load('{}.alpha.npy'.format(modelq_path))
    betaq = lda.get_topics()
    modelq = model.LDAEmBayes(alpha=alphaq, beta=betaq,
                              n_values=n_values)
    print(betaq.shape, betaq.min())
    return modelp, modelq, datar

# P, Q, ds = make_lda_prob(problem='stat.ME_math.PR_stat.TH',
#                          pname='math.PR', qname='stat.ME', rname='stat.TH')
# ns = [100, 200, 300]
ns = [100, 200, 300, 400, 500]
# P, Q, ds = make_lda_prob(problem='hep-ph_quant-ph_mixture',
#         pname='hep-ph_quant-ph_0.6', qname='hep-ph_quant-ph_0.7', rname='hep-ph')
P, Q, ds = make_lda_prob(problem='cs.LG_stat.ME_stat.TH',
        pname='cs.LG', qname='stat.ME', rname='stat.TH')

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
            # 'arxiv_AP_ME_TH':
            # ([100, 200, 300, ], ) + make_lda_prob(problem='stat.AP_stat.ME_stat.TH', 
            #                                     pname='stat.AP',
            #                                     qname='stat.ME', rname='stat.TH'),
            'arxiv_csLG_ME_TH':
            ([100, 200, 300, 400, 500], ) + make_lda_prob(problem='cs.LG_stat.ME_stat.TH', 
                                                pname='cs.LG',
                                                qname='stat.ME', rname='stat.TH'),
            # 'arxiv_csLG_ML_TH':
            # ([100, 200, 300,], ) + make_lda_prob(problem='cs.LG_stat.ML_stat.TH', 
            #                                     pname='cs.LG',
            #                                     qname='stat.ML', rname='stat.TH'),
            # 'arxiv_mathPR_statME_statTH':
            # ([100, 200, 300, 400, 500], ) + make_lda_prob(problem='stat.ME_math.PR_stat.TH', 
            #                                     pname='math.PR',
            #                                     qname='stat.ME', rname='stat.TH'),
            # 'arxiv_astro-ph.GA_astro-ph.SR_P06_Q07_Rastro-ph.SR':
            # ([100, 200, 300,], ) + make_lda_prob(problem='astro-ph.GA_astro-ph.SR_mixture',
            #                                     pname='astro-ph.SR_astro-ph.GA_0.6',
            #                                     qname='astro-ph.SR_astro-ph.GA_0.7', rname='astro-ph.SR'),
            # 'arxiv_hep-th_quant-ph_P06_Q07_Rquant-ph':
            # ([100, 200, 300,], ) + make_lda_prob(problem='hep-th_quant-ph_mixture',
            #                                     pname='quant-ph_hep-th_0.6',
            #                                     qname='quant-ph_hep-th_0.7', rname='quant-ph'),
            # 'arxiv_hep-ph_quant-ph_P06_Q07_Rhep-ph':
            # ([100, 200, 300,], ) + make_lda_prob(problem='hep-ph_quant-ph_mixture',
            #                                     pname='hep-ph_quant-ph_0.6',
            #                                     qname='hep-ph_quant-ph_0.7', rname='hep-ph'),
            # 'arxiv_cond-mat.mtrl-sci_cond-mat.mes-hall':
            # ([100, 200, 300,], ) + make_lda_prob(problem='cond-mat.mtrl-sci_cond-mat.mes-hall_mixture',
            #                                     pname='cond-mat.mtrl-sci_cond-mat.mes-hall_0.5',
            #                                     qname='cond-mat.mtrl-sci_cond-mat.mes-hall_0.7', rname='cond-mat.mtrl-sci'),

            }  # end of prob2tuples
    if prob_label not in prob2tuples:
        raise ValueError('Unknown problem label. Need to be one of %s'%str(list(prob2tuples.keys()) ))
    return prob2tuples[prob_label]


def run_problem(prob_label):
    """Run the experiment"""
    # ///////  submit jobs //////////
    # create folder name string
    #result_folder = glo.result_folder()
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
                    job = Ex6Job(SingleResultAggregator(), prob_label,
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
            # 'P': P, 'Q': Q,
            # 'data_source': ds, 
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

