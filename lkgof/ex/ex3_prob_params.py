"""Simulation to get rejection rates vs perturbation parameters"""

__author__ = 'noukoudashisoup'

import os 
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

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


def met_gmmd_med(P, Q, data_source, r):
    """
    Bounliphone et al., 2016's MMD-based 3-sample test.
    * Gaussian kernel. 
    * Gaussian width = mean of (median heuristic on (X, Z), median heuristic on
        (Y, Z))
    """
    if not P.has_datasource() or not Q.has_datasource():
        # Not applicable. Return {}.
        return {}

    # sample some data 
    datp, datq, datr = sample_pqr(P, Q, data_source, sample_size, r, only_from_r=False)

    # Start the timer here
    with util.ContextTimer() as t:
        X, Y, Z = datp.data(), datq.data(), datr.data()

        # hyperparameters of the test
        medxz = util.meddistance(np.vstack((X, Z)), subsample=1000)
        medyz = util.meddistance(np.vstack((Y, Z)), subsample=1000)
        medxyz = np.mean([medxz, medyz])
        k = kernel.KGauss(sigma2=medxyz**2)

        scmmd = SC_MMD(datp, datq, k, alpha=alpha)
        scmmd_result = scmmd.perform_test(datr)

    return {
            'test_result': scmmd_result, 'time_secs': t.secs
            }


def met_gksd_med(P, Q, data_source, r,
                 varest=util.second_order_ustat_variance_ustat,
                 ):
    """
    KSD-based model comparison test
        * One Gaussian kernel for the two statistics.
        * Requires exact marginals of the two models.
    """
    if not P.has_unnormalized_density() or not Q.has_unnormalized_density():
        # Not applicable. Return {}.
        return {}

    p = P.get_unnormalized_density()
    q = Q.get_unnormalized_density()
    # sample some data 
    datr = sample_pqr(None, None, data_source, sample_size, r, only_from_r=True)

    # Start the timer here
    with util.ContextTimer() as t:
        medz = util.meddistance(datr.data(), subsample=1000)
        k = kernel.KGauss(sigma2=medz**2)

        dcksd = mct.DC_KSD(p, q, k, k, seed=r+11, alpha=alpha,
                           varest=varest,
                           )
        dcksd_result = dcksd.perform_test(datr)

    return {
            'test_result': dcksd_result, 'time_secs': t.secs
            }


def met_glksd_med(P, Q, data_source, r, mc_sample=500,
                  varest=util.second_order_ustat_variance_ustat
                  ):
    """
    KSD-based model comparison test for latent variable models.
    - Use U-statistic variance estimator 
    """
    # sample some data 
    datr = sample_pqr(None, None, data_source, sample_size, r, only_from_r=True)

    # Start the timer here
    with util.ContextTimer() as t:
        medz = util.meddistance(datr.data(), subsample=1000)
        k = kernel.KGauss(sigma2=medz**2)

        n_burnin_p = burnin_sizes.get(type(P), 500)
        n_burnin_q = burnin_sizes.get(type(Q), 500)
        mc_param_p = MCParam(mc_sample, n_burnin_p)
        mc_param_q = MCParam(mc_sample, n_burnin_q)
        ldcksd = mct.LDC_KSD(P, Q, k, k,
                             seed=r+11, alpha=alpha,
                             mc_param_p=mc_param_p, mc_param_q=mc_param_q,
                             varest=varest,
        )
        dcksd_result = ldcksd.perform_test(datr)

    return {
            'test_result': dcksd_result, 'time_secs': t.secs
            }


def met_glksd_med_mc1000(P, Q, data_source, r, mc_sample=500):
    return met_glksd_med(P, Q, data_source, r, mc_sample=1000)


def met_glksd_med_vstat(P, Q, data_source, r):
    """
    LKSD model comparison test
        * One Gaussian kernel for the two statistics.
        * Markov chain sample size: 500
        * Use V-statistic variance estimator
    """

    return met_glksd_med(P, Q, data_source, r,
                         varest=util.second_order_ustat_variance_vstat,
                         )


def met_glksd_med_jackknife(P, Q, data_source, r):
    """
    LKSD model comparison test
        * One Gaussian kernel for the two statistics.
        * Markov chain sample size: 500
        * Use a jackknife variance estimator
    """

    return met_glksd_med(P, Q, data_source, r,
                         varest=util.second_order_ustat_variance_jackknife,
                         )


def met_dis_gbowlksd(P, Q, data_source, r):
    met_func = ex2.met_dis_gbowlksd
    return met_func(P, Q, data_source, sample_size, r)


def met_dis_gbowlksd_vstat(P, Q, data_source, r):
    met_func = ex2.met_dis_gbowlksd_vstat
    return met_func(P, Q, data_source, sample_size, r)


def met_dis_gbowlksd_jackknife(P, Q, data_source, r):
    met_func = ex2.met_dis_gbowlksd_jackknife
    return met_func(P, Q, data_source, sample_size, r)


def met_covimqlksd(P, Q, data_source, r, mc_sample=500,
                  varest=util.second_order_ustat_variance_jackknife,
                  ):
    """
    KSD-based model comparison test for latent variable models.
    - Use jackknife variance estimator 
    """
    # sample some data 
    datr = sample_pqr(None, None, data_source, sample_size, r, only_from_r=True)

    # Start the timer here
    with util.ContextTimer() as t:
        X = datr.data()
        X_ = X - X.mean(axis=0)
        cov = np.dot(X_.T, X_)
        k = kernel.KPIMQ(P=cov)

        n_burnin_p = burnin_sizes.get(type(P), 500)
        n_burnin_q = burnin_sizes.get(type(Q), 500)
        mc_param_p = MCParam(mc_sample, n_burnin_p)
        mc_param_q = MCParam(mc_sample, n_burnin_q)
        ldcksd = mct.LDC_KSD(P, Q, k, k,
                             seed=r+11, alpha=alpha,
                             mc_param_p=mc_param_p, mc_param_q=mc_param_q,
                             varest=varest,
        )
        ldcksd_result = ldcksd.perform_test(datr)

    return {
            'test_result': ldcksd_result, 'time_secs': t.secs
            }


def met_covimqmmd(P, Q, data_source, r):
    """
    Bounliphone et al., 2016's MMD-based 3-sample test.
    * Gaussian kernel. 
    * Gaussian width = mean of (median heuristic on (X, Z), median heuristic on
        (Y, Z))
    """
    if not P.has_datasource() or not Q.has_datasource():
        # Not applicable. Return {}.
        return {}

    ds_p = P.get_datasource()
    ds_q = Q.get_datasource()
    # sample some data 
    datp, datq, datr = sample_pqr(P, Q, data_source, sample_size, r, only_from_r=False)

    # Start the timer here
    with util.ContextTimer() as t:
        X, Y, Z = datp.data(), datq.data(), datr.data()
        X_ = X - X.mean(axis=0)
        cov = np.dot(X_.T, X_)
        k = kernel.KPIMQ(P=cov)

        # hyperparameters of the test
        scmmd = SC_MMD(datp, datq, k, alpha=alpha)
        scmmd_result = scmmd.perform_test(datr)

    return {
            'test_result': scmmd_result, 'time_secs': t.secs
            }


# Define our custom Job, which inherits from base class IndependentJob
class Ex3Job(IndependentJob):
   
    def __init__(self, aggregator, P, Q, data_source, prob_label, rep, met_func,
                 problem, param):
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
        self.problem = problem

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
        problem = self.problem

        logger.info("computing. %s. prob=%s, r=%d,\
                param=%g"%(met_func.__name__, prob_label, r, param))
        with util.ContextTimer() as t:
            job_result = met_func(P, Q, data_source, r)

            # create ScalarResult instance
            result = SingleResult(job_result)
            # submit the result to my own aggregator
            self.aggregator.submit_result(result)
            func_name = met_func.__name__

        logger.info("done. ex3: %s, prob=%s, r=%d, param=%g Took: %.3g s "%(func_name,
            prob_label, r, param, t.secs))

        # save result
        fname = '%s-%s-param%g_np%d_n%d_r%d_a%.3f.p' \
                %(prob_label, func_name, param, problem, sample_size, r, alpha)
        glo.ex_save_result(ex, job_result, prob_label, fname)

# This import is needed so that pickle knows about the class Ex3Job.
# pickle is used when collecting the results from the submitted jobs.
from lkgof.ex.ex3_prob_params import Ex3Job
from lkgof.ex.ex3_prob_params import met_gmmd_med
from lkgof.ex.ex3_prob_params import met_gksd_med
from lkgof.ex.ex3_prob_params import met_glksd_med
from lkgof.ex.ex3_prob_params import met_glksd_med_vstat
from lkgof.ex.ex3_prob_params import met_glksd_med_jackknife
from lkgof.ex.ex3_prob_params import met_glksd_med_mc1000
from lkgof.ex.ex3_prob_params import met_dis_gbowlksd
from lkgof.ex.ex3_prob_params import met_dis_gbowlksd_vstat
from lkgof.ex.ex3_prob_params import met_dis_gbowlksd_jackknife
from lkgof.ex.ex3_prob_params import met_covimqmmd
from lkgof.ex.ex3_prob_params import met_covimqlksd

#--- experimental setting -----
ex = 3

# significance level of the test
alpha = 0.05

# repetitions for each sample size 
reps = 100

# sample size 
sample_size = 100

# num of problems
num_problems = 10

# burnin size
burnin_sizes = {
    model.DPMIsoGaussBase: 1000,
    model.PPCA: 200,
    model.LDAEmBayes: 5000,
}

# Markov chain sample size 
n_mcsamples = 500

# tests to try
method_funcs = [ 
    # met_gmmd_med,
    # met_gksd_med,
    # met_glksd_med,
    # met_glksd_med_vstat,
    # met_glksd_med_jackknife,
    # met_dis_gbowlksd,
    # met_dis_gbowlksd_vstat,
    # met_dis_gbowlksd_jackknife,
    met_covimqmmd,
    met_covimqlksd,
   ]

# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting already exists.
is_rerun = False
# ---------------------------


from lkgof.ex.ex2_vary_n_disc import make_lda_prob
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
    Return [(params, P, Q, ds)], a (list of) lists of tuples of
    - param: perturbation parameter
    - P: a lkgof.model.LatentVariableModel representing the model P
    - Q: a lkgof.model.LatentVariableModel representing the model Q
    - ds: a DataSource. The DataSource generates sample from R.

    * (P, Q, ds) together specity a model comparison problem.
    """
    # vary the prior mean of P
    # isogdpm_ps = [1.0, 1.5, 2, 2.5, 3.5, 4.0, 4.5, 5.0, ]
    def isogdpm_ps(ptbq):
        return [ptbq + 0.1 * j 
                  for j in (list(range(-5, 0)) + list(range(1, 5+1)))]
                  # for j in range(-5, 5+1)]
    ppca_ps = [10**(-i) for i in range(9, 0, -1)]
    lda_ps = [10**(-i) for i in range(10, 0, -2)]
    # isogdpm_ps = [2, 2.25, 2.5, 2.75, 3.25, 3.5, 3.75, 4.0,]

    prob2tuples = { 
        'isogdpm_ms_dx10_tr10_q1':
            [
                [(ptbp,)+ make_dpm_isogauss_prob(tr_size=10, ptbp=ptbp, ptbq=1., seed=1000+i)
                 for ptbp in isogdpm_ps(1.)]
                for i in range(num_problems)
            ],
        'isogdpm_ms_dx10_tr5_q1':
            [
                [(ptbp,)+ make_dpm_isogauss_prob(tr_size=5, ptbp=ptbp, ptbq=1., seed=1000+i)
                 for ptbp in isogdpm_ps(1.)]
                for i in range(num_problems)
            ],
        'isogdpm_ms_dx10_tr15_q1':
            [
                [(ptbp,)+ make_dpm_isogauss_prob(tr_size=15, ptbp=ptbp, ptbq=1., seed=1000+i)
                 for ptbp in isogdpm_ps(1.)]
                for i in range(num_problems)
            ],
        'ppca_ws_dx10_h0_p0':
            [
                [(ptbq,)+ make_ppca_prob(dx=10, dz=5, var=1.,ptbp=0., ptbq=ptbq, seed=13+i)
                 for ptbq in ppca_ps]
                for i in range(num_problems)
            ],
        'ppca_ws_dx50_h0_p0':
            [
                [(ptbq,)+ make_ppca_prob(dx=50, dz=10, var=1.,ptbp=0., ptbq=ptbq, seed=13+i)
                 for ptbq in ppca_ps]
                for i in range(num_problems)
            ],
        'ppca_ws_dx50_h0_p1':
            [
                [(ptbq,)+ make_ppca_prob(dx=50, dz=10, var=1.,ptbp=1., ptbq=1.+ptbq, seed=13+i)
                 for ptbq in ppca_ps]
                for i in range(num_problems)
            ],
        'lda_as_dx50_h0_p1':
            [
                [(ptbq,)+ make_lda_prob(n_words=100, n_topics=3, vocab_size=50, ptb_p=1.,
                                        ptb_q=1.+ptbq, seed=13+i)
                 for ptbq in lda_ps]
                for i in range(num_problems)
            ],

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


    # num_problems x problem settings
    L = get_params_pqrsource(prob_label)
    params = L[0]
    params = list(params)
    # problems x repetitions x len(ns) x #methods
    aggregators = np.empty((num_problems, reps, len(params), n_methods ),
                           dtype=object)
    for pri in range(num_problems):
        params, ps, qs, data_sources, = zip(*L[pri])
        params = list(params)
        ps = list(ps)
        qs = list(qs)
        data_sources = list(data_sources)
        for r in range(reps):
            for pi, param in enumerate(params):
                for mi, f in enumerate(method_funcs):
                    # name used to save the result
                    func_name = f.__name__
                    fname = '%s-%s-param%g_np%d_n%d_r%d_a%.3f.p' \
                        %(prob_label, func_name, param, pri, sample_size, r, alpha)
                    if not is_rerun and glo.ex_file_exists(ex, prob_label, fname):
                        logger.info('%s exists. Load and return.'%fname)
                        job_result = glo.ex_load_result(ex, prob_label, fname)

                        sra = SingleResultAggregator()
                        sra.submit_result(SingleResult(job_result))
                        aggregators[pri, r, pi, mi] = sra
                    else:
                        # result not exists or rerun
                        job = Ex3Job(SingleResultAggregator(), ps[pi], qs[pi], data_sources[pi], prob_label,
                                r, f, pri, param)

                        agg = engine.submit_job(job)
                        aggregators[pri, r, pi, mi] = agg

    # let the engine finish its business
    logger.info("Wait for all call in engine")
    engine.wait_for_all()

    # ////// collect the results ///////////
    logger.info("Collecting results")
    job_results = np.empty((num_problems, reps, len(params), n_methods), dtype=object)
    for pri in range(num_problems):
        for r in range(reps):
            for pi, param in enumerate(params):
                for mi, f in enumerate(method_funcs):
                    logger.info("Collecting result (%s, problem=%d, r=%d, param=%g)" %
                            (f.__name__, pri, r, param))
                    # let the aggregator finalize things
                    aggregators[pri, r, pi, mi].finalize()

                    # aggregators[i].get_final_result() returns a SingleResult instance,
                    # which we need to extract the actual result
                    job_result = aggregators[pri, r, pi, mi].get_final_result().result
                    job_results[pri, r, pi, mi] = job_result

    #func_names = [f.__name__ for f in method_funcs]
    #func2labels = exglobal.get_func2label_map()
    #method_labels = [func2labels[f] for f in func_names if f in func2labels]

    # save results 
    results = {'job_results': job_results, 
            # 'ps': ps, 'qs': qs,
            # 'list_data_source': data_sources, 
            'alpha': alpha, 'repeats': reps, 'params': params,
            'method_funcs': method_funcs, 'prob_label': prob_label,
            'num_problems': num_problems,
            }
    
    # class name 
    # fname = 'ex%d-%s-me%d_rs%g_pmi%g_pma%d_n%d_a%.3f.p' \
        # %(ex, prob_label, n_methods, reps, min(params), max(params), sample_size, alpha,)
    fname = 'ex%d-%s-me%d_rs%g_np%d_pmi%g_pma%d_n%d_a%.3f.p' \
        %(ex, prob_label, n_methods, reps, num_problems, min(params), max(params), sample_size, alpha,)

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

