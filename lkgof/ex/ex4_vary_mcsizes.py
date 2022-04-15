"""Simulation to get the test power vs increasing MC/burn-in sample size"""

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


def sample_pqr(P, Q, ds_r, n, r, burnin_size=0, mcsize=0, only_from_r=False,  
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
        n_model_samples = n + burnin_size + mcsize
    datp = ds_p.sample(n_model_samples, seed=r+1000)
    datq = ds_q.sample(n_model_samples, seed=r+2000)
    return datp, datq, datr

# -------------------------------------------------------

def _met_lksd(P, Q, data_source, b, r, k, mcsize=1, ps_p=None, ps_q=None,
              varest=util.second_order_ustat_variance_jackknife):
    """
    Wrapper for LKSD model comparison test (relative test). 
    Different tests can be defined depending on the input kernel k.
    """
    # sample some data 
    datr = sample_pqr(None, None, data_source, sample_size, r, only_from_r=True)

    # Start the timer here
    with util.ContextTimer() as t:
        mc_param_p = MCParam(mcsize, b)
        mc_param_q = MCParam(mcsize, b)
        ldcksd = mct.LDC_KSD(P, Q, k, k, seed=r+11, alpha=alpha,
                             mc_param_p=mc_param_p, mc_param_q=mc_param_q,
                             ps_p=ps_p, ps_q=ps_q,
                             varest=varest,
                             )
        ldcksd_result = ldcksd.perform_test(datr)

    return {
            'test_result': ldcksd_result, 'time_secs': t.secs,
            }

def met_imqlksd_med_mc1(P, Q, data_source, b, r, mcsize=1):
    """
    KSD-based model comparison test
        * One Median-scaled IMQ kernel for the two statistics.
        * Use jackknife variance estimator
    """

    # sample some data 
    kernel_data = sample_pqr(None, None, data_source, n=kernel_datasize, r=0, only_from_r=True)
    X = kernel_data.data()
    d = X.shape[1]
    dz = P.dim_l
    medX = util.meddistance(X)
    k = kernel.KPIMQ(medX**2 * np.eye(d))

    if type(P) is model.PPCA:
        bias = 1.
        def ps_p(X, n_sample, seed, n_burnin=1):
            with util.NumpySeedContext(seed+133):
                Z_init = np.random.randn(sample_size, dz) + bias
            return P.posterior(X, n_sample, seed=seed, n_burnin=n_burnin,
                               init_params=Z_init)
    ldcksd_result = _met_lksd(P, Q, data_source, b, r, k=k, mcsize=mcsize, ps_p=ps_p)
    return ldcksd_result

def met_imqlksd_med_mc10(P, Q, data_source, b, r):
    return met_imqlksd_med_mc1(P, Q, data_source, b, r, mcsize=10)

def met_imqlksd_med_mc100(P, Q, data_source, b, r):
    return met_imqlksd_med_mc1(P, Q, data_source, b, r, mcsize=100)

def met_imqlksd_med_mc1000(P, Q, data_source, b, r):
    return met_imqlksd_med_mc1(P, Q, data_source, b, r, mcsize=1000)

def met_imqlksd_med_mc10000(P, Q, data_source, b, r):
    return met_imqlksd_med_mc1(P, Q, data_source, b, r, mcsize=10000)


def met_imqlksd_med_mala_mc1(P, Q, data_source, b, r, mcsize=1):
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
    assert P.dim_l == Q.dim_l
    dz = P.dim_l 
    
    # stepsize = dz**(-1./3) * 1e-3
    stepsize = dz**(-1./3) * 1e-4
    bias = 1.
    def ps_p(X, n_sample, seed, n_burnin=1):
        with util.NumpySeedContext(seed+133):
            Z_init = np.random.randn(sample_size, dz) + bias
        return mcmc.ppca_mala(X, n_sample, Z_init, ppca_model=P,
                              stepsize=stepsize, seed=seed, n_burnin=n_burnin)
    def ps_q(X, n_sample, seed, n_burnin=1):
        with util.NumpySeedContext(seed+134):
            Z_init = np.random.randn(sample_size, dz)
        return mcmc.ppca_mala(X, n_sample, Z_init, ppca_model=Q,
                              stepsize=stepsize, seed=seed, n_burnin=n_burnin)

    ldcksd_result = _met_lksd(P, Q, data_source, b, r, k=k, ps_p=ps_p, ps_q=None, mcsize=mcsize)
    return ldcksd_result

def met_imqlksd_med_mala_mc10(P, Q, data_source, b, r):
    return met_imqlksd_med_mala_mc1(P, Q, data_source, b, r, mcsize=10)

def met_imqlksd_med_mala_mc100(P, Q, data_source, b, r):
    return met_imqlksd_med_mala_mc1(P, Q, data_source, b, r, mcsize=100)

def met_imqlksd_med_mala_mc1000(P, Q, data_source, b, r):
    return met_imqlksd_med_mala_mc1(P, Q, data_source, b, r, mcsize=1000)


def met_dis_imqbowlksd_mc1(P, Q, data_source, b, r, mcsize=1):
    """
    Latent KSD-based model comparison test (relative test).
        * IMQ BoW kernel for discrete observations.
    """
    if not np.all(P.n_values == Q.n_values):
        raise ValueError('P, Q have different domains. P.n_values = {}, Q.n_values = {}'.format(P.n_values, Q.n_values))
    n_values = P.n_values
    d = P.dim
    k = kernel.KIMQBoW(n_values, d, s2=1)

    ldcksd_result = _met_lksd(P, Q, data_source, b, r, k=k, mcsize=mcsize)
    return ldcksd_result

def met_dis_imqbowlksd_mc10(P, Q, data_source, b, r):
    return met_dis_imqbowlksd_mc1(P, Q, data_source, b, r, mcsize=10)

def met_dis_imqbowlksd_mc100(P, Q, data_source, b, r):
    return met_dis_imqbowlksd_mc1(P, Q, data_source, b, r, mcsize=100)

def met_dis_imqbowlksd_mc1000(P, Q, data_source, b, r):
    return met_dis_imqbowlksd_mc1(P, Q, data_source, b, r, mcsize=1000)

def met_dis_imqbowlksd_mc10000(P, Q, data_source, b, r):
    return met_dis_imqbowlksd_mc1(P, Q, data_source, b, r, mcsize=10000)


# Define our custom Job, which inherits from base class IndependentJob
class Ex4Job(IndependentJob):
   
    def __init__(self, aggregator, P, Q, data_source, prob_label, rep, met_func, b):
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
        self.b = b

    # we need to define the abstract compute method. It has to return an instance
    # of JobResult base class
    def compute(self):

        P = self.P
        Q = self.Q
        data_source = self.data_source 
        r = self.rep
        b = self.b
        met_func = self.met_func
        prob_label = self.prob_label

        logger.info("computing. %s. prob=%s, r=%d,\
                burn-in=%d"%(met_func.__name__, prob_label, r, b))
        with util.ContextTimer() as t:
            job_result = met_func(P, Q, data_source, b, r)

            # create ScalarResult instance
            result = SingleResult(job_result)
            # submit the result to my own aggregator
            self.aggregator.submit_result(result)
            func_name = met_func.__name__

        logger.info("done. ex4: %s, prob=%s, n=%d, r=%d, burn-in=%d. Took: %.3g s "%(func_name,
            prob_label, sample_size, r, b, t.secs))

        # save result
        fname = '%s-%s-n%d_b%d_r%d_a%.3f.p' \
                %(prob_label, func_name, sample_size, b, r, alpha )
        glo.ex_save_result(ex, job_result, prob_label, fname)

# This import is needed so that pickle knows about the class Ex1Job.
# pickle is used when collecting the results from the submitted jobs.
from lkgof.ex.ex4_vary_mcsizes import Ex4Job
from lkgof.ex.ex4_vary_mcsizes import _met_lksd
from lkgof.ex.ex4_vary_mcsizes import met_imqlksd_med_mc1
from lkgof.ex.ex4_vary_mcsizes import met_imqlksd_med_mc10
from lkgof.ex.ex4_vary_mcsizes import met_imqlksd_med_mc100
from lkgof.ex.ex4_vary_mcsizes import met_imqlksd_med_mc1000
from lkgof.ex.ex4_vary_mcsizes import met_imqlksd_med_mc10000
from lkgof.ex.ex4_vary_mcsizes import met_imqlksd_med_mala_mc1
from lkgof.ex.ex4_vary_mcsizes import met_imqlksd_med_mala_mc10
from lkgof.ex.ex4_vary_mcsizes import met_imqlksd_med_mala_mc100
from lkgof.ex.ex4_vary_mcsizes import met_imqlksd_med_mala_mc1000
from lkgof.ex.ex4_vary_mcsizes import met_dis_imqbowlksd_mc1
from lkgof.ex.ex4_vary_mcsizes import met_dis_imqbowlksd_mc10
from lkgof.ex.ex4_vary_mcsizes import met_dis_imqbowlksd_mc100
from lkgof.ex.ex4_vary_mcsizes import met_dis_imqbowlksd_mc1000
from lkgof.ex.ex4_vary_mcsizes import met_dis_imqbowlksd_mc10000

from lkgof.ex.ex2_vary_n_disc import make_lda_prob
from lkgof.ex.ex1_vary_n import make_ppca_prob
from lkgof.ex.ex3_prob_params import make_dpm_isogauss_prob



#--- experimental setting -----
ex = 4

# significance level of the test
alpha = 0.05

# repetitions for each sample size 
reps = 300

# kernel data size
kernel_datasize = 1000

# sample size 
sample_size = 100

# tests to try
method_funcs = [ 
    # met_imqlksd_med_mc1,
    # met_imqlksd_med_mc10,
    # met_imqlksd_med_mc100,
    # met_imqlksd_med_mc1000,
    met_imqlksd_med_mala_mc1,
    met_imqlksd_med_mala_mc10,
    met_imqlksd_med_mala_mc100,
    met_imqlksd_med_mala_mc1000,
    # met_dis_imqbowlksd_mc1,
    # met_dis_imqbowlksd_mc10,
    # met_dis_imqbowlksd_mc100,
    # met_dis_imqbowlksd_mc1000,
    # met_dis_imqbowlksd_mc10000,
   ]

# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting already exists.
is_rerun = False
# ---------------------------


def get_bs_pqrsource(prob_label):
    """
    Return (ns, P, Q, ds), a tuple of
    - bs: a list of burn-in sizes b's
    - P: a lkgof.model.LatentVariableModel representing the model P
    - Q: a lkgof.model.LatentVariableModel representing the model Q
    - ds: a DataSource. The DataSource generates sample from R.

    * (P, Q, ds) together specity a three-sample (or model comparison) problem.
    """
    ppca_burnins = [50*i for i in range(1, 12+1)]
    lda_burnins = [500*i for i in range(1, 8+1)]
    gdpm_burnins = [200*i for i in range(1, 9+1)]

    prob2tuples = { 
        'ppca_h0_dx100_dz10_p1_q1+1e-5':
            # list of sample sizes
            (ppca_burnins, ) + make_ppca_prob(dx=100, dz=10, ptbp=1., ptbq=1+1e-5),
        'ppca_h1_dx100_dz10_p2_q1':
            # list of sample sizes
            (ppca_burnins, ) + make_ppca_prob(dx=100, dz=10, ptbp=2., ptbq=1),
        'lda_h0_dx50_v1000_t3_p05q1temp1e-1':
            (lda_burnins, ) + make_lda_prob(n_words=50, n_topics=3,
                                            vocab_size=1000,
                                            ptb_p=0.5, ptb_q=1., temp=0.1),
        'lda_h0_dx50_v1000_t3_p05q1temp1':
            (lda_burnins, ) + make_lda_prob(n_words=50, n_topics=3,
                                            vocab_size=1000,
                                            ptb_p=0.5, ptb_q=1., temp=1),
        'lda_h1_dx50_v1000_t3_p1q05temp1e-1':
            (lda_burnins, ) + make_lda_prob(n_words=50, n_topics=3,
                                            vocab_size=1000,
                                            ptb_p=1.0, ptb_q=0.5, temp=0.1),
        'lda_h1_dx50_v1000_t3_p1q05temp1':
            (lda_burnins, ) + make_lda_prob(n_words=50, n_topics=3,
                                            vocab_size=1000,
                                            ptb_p=1.0, ptb_q=0.5, temp=1),
        'lda_h0_dx50_v10000_t3_p05q1temp1':
            (lda_burnins, ) + make_lda_prob(n_words=50, n_topics=3,
                                            vocab_size=10000,
                                            ptb_p=0.5, ptb_q=1., temp=1),

        'lda_h1_dx50_v10000_t3_p1q05temp1':
            (lda_burnins, ) + make_lda_prob(n_words=50, n_topics=3,
                                            vocab_size=10000,
                                            ptb_p=1.0, ptb_q=0.5, temp=1),
        'lda_h1_dx50_v10000_t3_p1q05temp1e-1':
            (lda_burnins, ) + make_lda_prob(n_words=50, n_topics=3,
                                            vocab_size=10000,
                                            ptb_p=1.0, ptb_q=0.5, temp=1e-1),
        'isogdpm_h1_dx10_tr5_p12e-1_q1':
            # list of sample sizes
            (gdpm_burnins, ) + make_dpm_isogauss_prob(tr_size=5, dx=10, ptbp=1.2, ptbq=1.),
        'isogdpm_h0_dx10_tr5_p9e-1_q1':
            # list of sample sizes
            (gdpm_burnins, ) + make_dpm_isogauss_prob(tr_size=5, dx=10, ptbp=0.9, ptbq=1.),
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
    bs, P, Q, ds, = get_bs_pqrsource(prob_label)

    # repetitions x len(bs) x #methods
    aggregators = np.empty((reps, len(bs), n_methods ), dtype=object)

    for r in range(reps):
        for bi, b in enumerate(bs):
            for mi, f in enumerate(method_funcs):
                # name used to save the result
                func_name = f.__name__
                fname = '%s-%s-n%d_b%d_r%d_a%.3f.p' \
                        %(prob_label, func_name, sample_size, b, r, alpha,)
                if not is_rerun and glo.ex_file_exists(ex, prob_label, fname):
                    logger.info('%s exists. Load and return.'%fname)
                    job_result = glo.ex_load_result(ex, prob_label, fname)

                    sra = SingleResultAggregator()
                    sra.submit_result(SingleResult(job_result))
                    aggregators[r, bi, mi] = sra
                else:
                    # result not exists or rerun
                    job = Ex4Job(SingleResultAggregator(), P, Q, ds, prob_label,
                            r, f, b)

                    agg = engine.submit_job(job)
                    aggregators[r, bi, mi] = agg

    # let the engine finish its business
    logger.info("Wait for all call in engine")
    engine.wait_for_all()

    # ////// collect the results ///////////
    logger.info("Collecting results")
    job_results = np.empty((reps, len(bs), n_methods), dtype=object)
    for r in range(reps):
        for bi, b in enumerate(bs):
            for mi, f in enumerate(method_funcs):
                logger.info("Collecting result (%s, n=%d, r=%d, burn-in=%d)" %
                        (f.__name__, sample_size, r, b))
                # let the aggregator finalize things
                aggregators[r, bi, mi].finalize()

                # aggregators[i].get_final_result() returns a SingleResult instance,
                # which we need to extract the actual result
                job_result = aggregators[r, bi, mi].get_final_result().result
                job_results[r, bi, mi] = job_result

    #func_names = [f.__name__ for f in method_funcs]
    #func2labels = exglobal.get_func2label_map()
    #method_labels = [func2labels[f] for f in func_names if f in func2labels]

    # save results 
    results = {'job_results': job_results, 
            'P': P, 'Q': Q,
            'data_source': ds, 'sample_size': sample_size,
            'alpha': alpha, 'repeats': reps, 'bs': bs,
            'method_funcs': method_funcs, 'prob_label': prob_label,
            }
    
    # class name 
    fname = 'ex%d-%s-me%d_rs%d_bmi%d_bma%d_n%d_a%.3f.p' \
        %(ex, prob_label, n_methods, reps, min(bs), max(bs), sample_size, alpha,)

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

