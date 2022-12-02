"""Simulation to get rejection rates vs perturbation parameters"""

__author__ = 'noukoudashisoup'

import os 
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

import lkgof.ex.ex1_vary_n as ex1
import lkgof.ex.ex2_vary_n_disc as ex2
import lkgof.glo as glo
import lkgof.model as model
import lkgof.data as data
# goodness-of-fit test
import lkgof.kernel as kernel
import lkgof.util as util
import lkgof.mctest as mct
from lkgof.mctest import SC_MMD
from lkgof.goftest import MCParam
from itertools import product


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
    - P: a lkgof.model.LatentVariableModel (candidate model 1)
    - Q: a lkgof.model.LatentVariableModel (candidate model 2)
    - data_source: a kgof.data.DataSource for generating the data (i.e., draws
          from R)
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


def met_gmmd_med(P, Q, data_source, r):
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
    scmmd_result = _met_mmd(P, Q, data_source, sample_size, r, k=k,)
    return scmmd_result


def met_imqmmd_cov(P, Q, data_source, r):
    """
    Bounliphone et al., 2016's MMD-based 3-sample test.
    * Precondition IMQ kernel with sample covariance preconditioner
    """
    # sample some data 
    kernel_data = sample_pqr(None, None, data_source, n=kernel_datasize, r=0, only_from_r=True)
    X = kernel_data.data()

    cov = np.cov(X, rowvar=False)
    k = kernel.KPIMQ(P=cov)
    scmmd_result = _met_mmd(P, Q, data_source, sample_size, r, k=k)
    return scmmd_result


def met_imqmmd_med(P, Q, data_source, r):
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

    scmmd_result = _met_mmd(P, Q, data_source, sample_size, r, k=k)
    return scmmd_result


def _met_lksd(P, Q, data_source, n, r, k,
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
                             varest=varest,
                             )
        ldcksd_result = ldcksd.perform_test(datr)

    return {
            'test_result': ldcksd_result, 'time_secs': t.secs,
            }


def met_glksd_med(P, Q, data_source, r):
    """
    LKSD model comparison test
        * One Gaussian kernel for the two statistics.
        * Use jackknife variance estimator
    """
    # sample some data 
    kernel_data = sample_pqr(None, None, data_source, n=kernel_datasize, r=0, only_from_r=True)
    medz = util.meddistance(kernel_data.data(), subsample=1000)
    k = kernel.KGauss(sigma2=medz**2)

    ldcksd_result =  _met_lksd(P, Q, data_source, sample_size, r, k=k, )
    return ldcksd_result
 

def met_imqlksd_med(P, Q, data_source, r, 
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

    ldcksd_result = _met_lksd(P, Q, data_source, sample_size, r, k=k, varest=varest)
    return ldcksd_result


def met_imqlksd_med_ustatvar(P, Q, data_source, r):
    """
    KSD-based model comparison test
        * One Median-scaled IMQ kernel for the two statistics.
        * Use U-stat variance estimator
    """

    # sample some data 
    varest = util.second_order_ustat_variance_ustat
    return met_imqlksd_med(P, Q, data_source, r, varest=varest)


def met_imqlksd_med_vstatvar(P, Q, data_source, r):
    """
    KSD-based model comparison test
        * One Median-scaled IMQ kernel for the two statistics.
        * Use V-stat variance estimator
    """

    # sample some data 
    varest = util.second_order_ustat_variance_vstat
    return met_imqlksd_med(P, Q, data_source, r, varest=varest)


def met_imqlksd_cov(P, Q, data_source, r):
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

    ldcksd_result = _met_lksd(P, Q, data_source, sample_size, r, k=k,)

    return ldcksd_result

def met_dis_imqbowmmd(P, Q, data_source, r):
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
    result = _met_mmd(P, Q, data_source, sample_size, r, k)
    return result


def met_dis_imqbowlksd(P, Q, data_source, r, 
                       varest=util.second_order_ustat_variance_jackknife):
    """
    Latent KSD-based model comparison test (relative test).
        * IMQ BoW kernel for discrete observations.
        * Use jackknife variance estimator
    """
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim

    k = kernel.KIMQBoW(n_values, d, s2=1)
    result = _met_lksd(P, Q, data_source, sample_size, r, k, varest=varest)
    return result

def met_dis_imqbowlksd_ustatvar(P, Q, data_source, r):
    varest = util.second_order_ustat_variance_ustat
    return met_imqlksd_med(P, Q, data_source, r, varest=varest)


def met_dis_imqbowlksd_vstatvar(P, Q, data_source, r):
    varest = util.second_order_ustat_variance_vstat
    return met_imqlksd_med(P, Q, data_source, r, varest=varest)


def met_dis_imqbow_mflksd(P, Q, data_source, r,
                          varest=util.second_order_ustat_variance_jackknife,
    ):
    """
    Latent MFKSD-based model comparison test (relative test).
        * IMQ BoW kernel for discrete observations.
        * Use jackknife variance estimator
    """
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim

    k = kernel.KIMQBoW(n_values, d, s2=1)
    # sample some data 
    datr = sample_pqr(None, None, data_source, sample_size, r, only_from_r=True)

    # Start the timer here
    n_mc = 1000
    with util.ContextTimer() as t:
        n_burnin_p = 500
        n_burnin_q = 500
        mc_param_p = MCParam(n_mc, n_burnin_p)
        mc_param_q = MCParam(n_mc, n_burnin_q)
        ldcksd = mct.LDC_MFKSD(P, Q, k, k, seed=r+11, alpha=alpha,
                               mc_param_p=mc_param_p, mc_param_q=mc_param_q,
                               varest=varest,
                             )
        ldcksd_result = ldcksd.perform_test(datr)
    return {
            'test_result': ldcksd_result, 'time_secs': t.secs,
            }


# Define our custom Job, which inherits from base class IndependentJob
class Ex7Job(IndependentJob):
   
    def __init__(self, aggregator, P, Q, data_source, prob_label, rep, met_func,
                 param):
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
        self.param = param

    # we need to define the abstract compute method. It has to return an instance
    # of JobResult base class
    def compute(self):

        P = self.P
        Q = self.Q
        data_source = self.data_source 
        r = self.rep
        param = self.param
        met_func = self.met_func
        prob_label = self.prob_label

        logger.info("computing. %s. prob=%s, r=%d,\
                param=%s"%(met_func.__name__, prob_label, r, param))
        with util.ContextTimer() as t:
            job_result = met_func(P, Q, data_source, r)

            # create ScalarResult instance
            result = SingleResult(job_result)
            # submit the result to my own aggregator
            self.aggregator.submit_result(result)
            func_name = met_func.__name__

        logger.info("done. ex7: %s, prob=%s, r=%d, param=%s Took: %.3g s "%(func_name,
            prob_label, r, param, t.secs))

        # save result
        fname = '%s-%s-param_p%g_%g_n%d_mc%d_r%d_a%.3f.p' \
                        %(prob_label, func_name, param[0], param[1],  sample_size, n_mcsamples, r, alpha)

        glo.ex_save_result(ex, job_result, prob_label, fname)

# This import is needed so that pickle knows about the class Ex7Job.
# pickle is used when collecting the results from the submitted jobs.
from lkgof.ex.ex7_prob_params_heat import Ex7Job
from lkgof.ex.ex7_prob_params_heat import _met_mmd
from lkgof.ex.ex7_prob_params_heat import _met_lksd
from lkgof.ex.ex7_prob_params_heat import met_gmmd_med
from lkgof.ex.ex7_prob_params_heat import met_imqmmd_med
from lkgof.ex.ex7_prob_params_heat import met_imqmmd_cov
from lkgof.ex.ex7_prob_params_heat import met_glksd_med
from lkgof.ex.ex7_prob_params_heat import met_imqlksd_cov
from lkgof.ex.ex7_prob_params_heat import met_imqlksd_med
from lkgof.ex.ex7_prob_params_heat import met_imqlksd_med_ustatvar
from lkgof.ex.ex7_prob_params_heat import met_imqlksd_med_vstatvar
from lkgof.ex.ex7_prob_params_heat import met_dis_imqbowlksd
from lkgof.ex.ex7_prob_params_heat import met_dis_imqbowlksd_ustatvar
from lkgof.ex.ex7_prob_params_heat import met_dis_imqbowlksd_vstatvar
from lkgof.ex.ex7_prob_params_heat import met_dis_imqbowmmd
from lkgof.ex.ex7_prob_params_heat import met_dis_imqbow_mflksd

#--- experimental setting -----
ex = 7

# significance level of the test
alpha = 0.05

# repetitions for each sample size 
reps = 100

# sample size 
sample_size = 300

# kernel data size
kernel_datasize = 500

# burnin size
burnin_sizes = {
    model.DPMIsoGaussBase: 1000,
    model.PPCA: 200,
    model.LDAEmBayes: 4000,
}

# Markov chain sample size 
n_mcsamples = 1000

# tests to try
method_funcs = [ 
    # met_gmmd_med,
    # met_glksd_med,
    # met_imqlksd_med,
    # met_imqlksd_med_ustatvar,
    # met_imqlksd_med_vstatvar,
    # met_imqmmd_med,
    # met_imqmmd_cov,
    # met_imqlksd_cov,
    met_dis_imqbowlksd,
    met_dis_imqbowmmd,
    # met_dis_imqbow_mflksd,
    # met_dis_imqbowlksd_ustatvar,
    # met_dis_imqbowlksd_vstatvar,
   ]

# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting already exists.
is_rerun = False
# ---------------------------


from lkgof.ex.ex2_vary_n_disc import make_lda_prob
from lkgof.ex.ex2_vary_n_disc import make_lda_mix_prob
from lkgof.ex.ex1_vary_n import make_ppca_prob


def make_dpm_isogauss_prob(tr_size=10, dx=10, var=2., prior_var=1.,
                           ptbp=1., ptbq=2., seed=1045):
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

    mean = np.zeros([dx])  # same as prior mean
    ds_var = var + prior_var
    ds = data.DSIsotropicNormal(mean, ds_var)
    obs = ds.sample(tr_size, seed=seed).data()
    
    ptb_dir = np.ones([dx]) / dx**0.5
    modelp = model.DPMIsoGaussBase(var, mean+ptb_dir*ptbp, prior_var, obs=obs)
    modelq = model.DPMIsoGaussBase(var, mean+ptb_dir*ptbq, prior_var, obs=obs)
    return modelp, modelq, ds


def get_params_pqrsource(prob_label):
    """
    Return [(params, P, Q, ds)...], ptb_range) tuple of the following two things:
    1. A list of problems, where (P, Q, ds) together specity a model comparison problem: 
        - param: tuple of two perturbation parameters
        - P: a lkgof.model.LatentVariableModel representing the model P
        - Q: a lkgof.model.LatentVariableModel representing the model Q
        - ds: a DataSource. The DataSource generates sample from R.
    2. Perturbation parameters
    """
    # vary the prior mean of P
    def isogdpm_ps(ptbq):
        return [ptbq + 0.1 * j 
                  for j in (list(range(-5, 0)) + list(range(1, 5+1)))]

    ppca_ps = np.logspace(-2, 0.5, 10)+1
    lda_ps = np.logspace(-2, 0, 10)+0.5

    prob2tuples = { 
        'ppca_ws_dx100_dz10':
            (
                [((ptbp, ptbq),)+ make_ppca_prob(dx=100, dz=10, var=1., 
                                        ptbp=ptbp, ptbq=ptbq, seed=13)
                 for ptbp, ptbq in product(ppca_ps, ppca_ps)],
                ppca_ps,
            ),
        'lda_as_dx50_v10000_t3_temp1':
            (
                [((ptbp, ptbq),)+ make_lda_prob(n_words=50, n_topics=3, vocab_size=10000, ptb_p=ptbp,
                                        ptb_q=ptbq, seed=13, temp=1.)
                 for ptbp, ptbq in product(lda_ps, lda_ps)],
                lda_ps,
            ),
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
    #engine = SerialComputationEngine()
    partitions = expr_configs['slurm_partitions']
    if partitions is None:
        engine = SlurmComputationEngine(batch_parameters)
    else:
        engine = SlurmComputationEngine(batch_parameters, partition=partitions)
    n_methods = len(method_funcs)


    L, ptb_range = get_params_pqrsource(prob_label)
    num_ptbs = len(ptb_range)
    # repetitions x ptbs for P x ptbs for Q x #methods
    aggregators = np.empty((reps, num_ptbs, num_ptbs, n_methods ),
                           dtype=object)
    params, ps, qs, data_sources, = zip(*L)
    params = list(params)
    ps = list(ps)
    qs = list(qs)
    data_sources = list(data_sources)
    indices = [(i, j) for i, j in product(range(num_ptbs), range(num_ptbs))]
    for r in range(reps):
        for pi, param in enumerate(params):
            for mi, f in enumerate(method_funcs):
                # name used to save the result
                func_name = f.__name__
                ptbp, ptbq = param
                fname = '%s-%s-param_p%g_%g_n%d_mc%d_r%d_a%.3f.p' \
                    %(prob_label, func_name, ptbp, ptbq, sample_size, n_mcsamples, r, alpha)
                idx_p, idx_q = indices[pi]
                assert ptbp == ptb_range[idx_p]
                assert ptbq == ptb_range[idx_q]
                if not is_rerun and glo.ex_file_exists(ex, prob_label, fname):
                    logger.info('%s exists. Load and return.'%fname)
                    job_result = glo.ex_load_result(ex, prob_label, fname)

                    sra = SingleResultAggregator()
                    sra.submit_result(SingleResult(job_result))
                    aggregators[r, idx_p, idx_q, mi] = sra
                else:
                    # result not exists or rerun
                    job = Ex7Job(SingleResultAggregator(), ps[pi], qs[pi], data_sources[pi], prob_label,
                            r, f, param)

                    agg = engine.submit_job(job)
                    aggregators[r, idx_p, idx_q, mi] = agg

    # let the engine finish its business
    logger.info("Wait for all call in engine")
    engine.wait_for_all()

    # ////// collect the results ///////////
    logger.info("Collecting results")
    job_results = np.empty((reps, num_ptbs, num_ptbs, n_methods), dtype=object)
    for r in range(reps):
        for pi, param in enumerate(params):
            for mi, f in enumerate(method_funcs):
                idx_p, idx_q = indices[pi]
                logger.info("Collecting result (%s, r=%d, param=%s)" %
                        (f.__name__, r, param))
                # let the aggregator finalize things
                aggregators[r, idx_p, idx_q, mi].finalize()

                # aggregators[i].get_final_result() returns a SingleResult instance,
                # which we need to extract the actual result
                job_result = aggregators[r, idx_p, idx_q, mi].get_final_result().result
                job_results[r, idx_p, idx_q, mi] = job_result

    #func_names = [f.__name__ for f in method_funcs]
    #func2labels = exglobal.get_func2label_map()
    #method_labels = [func2labels[f] for f in func_names if f in func2labels]

    # save results 
    results = {'job_results': job_results, 
            # 'ps': ps, 'qs': qs,
            # 'list_data_source': data_sources, 
            'alpha': alpha, 'repeats': reps, 'ptbs': ptb_range, 
            'method_funcs': method_funcs, 'prob_label': prob_label,
            }
    
    # class name 
    # fname = 'ex%d-%s-me%d_rs%g_pmi%g_pma%d_n%d_a%.3f.p' \
        # %(ex, prob_label, n_methods, reps, min(params), max(params), sample_size, alpha,)

    fname = 'ex%d-%s-me%d_rs%g_pmi%g_pma%g_n%d_mc%d_a%.3f.p' \
        %(ex, prob_label, n_methods, reps, 
          min(ptb_range), max(ptb_range), sample_size, n_mcsamples, alpha,)


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

