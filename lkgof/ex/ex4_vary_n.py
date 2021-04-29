"""Simulation to get the test power vs increasing sample size"""

__author__ = 'wittawat'

# Two-sample test
import kmod
import lkgof
from kmod import data, util
from kmod import mctest as mct
import lkgof.glo as glo
import lkgof.mctest as mct
import kmod.model as model
import lkgof.model
import lkgof.density as density
# goodness-of-fit test
import lkgof.goftest as gof
import lkgof.kernel as kernel

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
import math
import os
import sys 

"""
All the method functions (starting with met_) return a dictionary with the
following keys:
    - test: test object. (may or may not return to save memory)
    - test_result: the result from calling perform_test(te).
    - time_secs: run time in seconds 

    * A method function may return an empty dictionary {} if the inputs are not
    applicable. For example, if density functions are not available, but the
    method function is FSSD which needs them.

All the method functions take the following mandatory inputs:
    - P: a kmod.model.Model (candidate model 1)
    - Q: a kmod.model.Model (candidate model 2)
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
    datr = ds_r.sample(n, seed=r)
    if only_from_r:
        return datr
    datp = ds_p.sample(n, seed=r+10000)
    datq = ds_q.sample(n, seed=r+20000)
    return datp, datq, datr

#-------------------------------------------------------
def met_gmmd_med(P, Q, data_source, n, r):
    """
    Use met_gmmd_med_bounliphone(). It uses the median heuristic following
    Bounliphone et al., 2016.

    Bounliphone et al., 2016's MMD-based 3-sample test.
    * Gaussian kernel. 
    * Gaussian width = mean of (median heuristic on (X, Z), median heuristic on
        (Y, Z))
    * Use full sample for testing (no
    holding out for optimization)
    """
    if not P.has_datasource() or not Q.has_datasource():
        # Not applicable. Return {}.
        return {}

    ds_p = P.get_datasource()
    ds_q = Q.get_datasource()
    # sample some data 
    datp, datq, datr = sample_pqr(ds_p, ds_q, data_source, n, r, only_from_r=False)

    # Start the timer here
    with util.ContextTimer() as t:
        X, Y, Z = datp.data(), datq.data(), datr.data()

        # hyperparameters of the test
        medxz = util.meddistance(np.vstack((X, Z)), subsample=1000)
        medyz = util.meddistance(np.vstack((Y, Z)), subsample=1000)
        medxyz = np.mean([medxz, medyz])
        k = kernel.KGaussBumpL2(sigma2=medxyz**2, r=radius, scale=scale)

        scmmd = mct.SC_MMD(datp, datq, k, alpha=alpha)
        scmmd_result = scmmd.perform_test(datr)

    return {
            # This key "test" can be removed.             
            #'test': scmmd, 
            'test_result': scmmd_result, 'time_secs': t.secs}

def met_gmmd_med_bounliphone(P, Q, data_source, n, r):
    """
    Bounliphone et al., 2016's MMD-based 3-sample test.
    * Gaussian kernel. 
    * Gaussian width = chosen as described in https://github.com/wbounliphone/relative_similarity_test/blob/4884786aa3fe0f41b3ee76c9587de535a6294aee/relativeSimilarityTest_finalversion.m 
    * Use full sample for testing (no
    holding out for optimization)
    """
    if not P.has_datasource() or not Q.has_datasource():
        # Not applicable. Return {}.
        return {}


    ds_p = P.get_datasource()
    ds_q = Q.get_datasource()
    # sample some data 
    datp, datq, datr = sample_pqr(ds_p, ds_q, data_source, n, r, only_from_r=False)

    # Start the timer here
    with util.ContextTimer() as t:
        X, Y, Z = datp.data(), datq.data(), datr.data()

        med2 = mct.SC_MMD.median_heuristic_bounliphone(X, Y, Z, subsample=1000, seed=r+3)
        k = kernel.KGaussBumpL2(sigma2=med2, r=radius, scale=scale)

        scmmd = mct.SC_MMD(datp, datq, k, alpha=alpha)
        scmmd_result = scmmd.perform_test(datr)

    return {
            # This key "test" can be removed.             
            # 'test': scmmd, 
            'test_result': scmmd_result, 'time_secs': t.secs}


def met_gksd_med(P, Q, data_source, n, r):
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
    datr = sample_pqr(None, None, data_source, n, r, only_from_r=True)

    # Start the timer here
    with util.ContextTimer() as t:
        medz = util.meddistance(datr.data(), subsample=1000)
        k = kernel.KGaussBumpL2(sigma2=medz**2, r=radius, scale=scale)

        dcksd= mct.DC_KSD(p, q, k, k, seed=r+11, alpha=alpha)
        dcksd_result = dcksd.perform_test(datr)

    return {
            # This key "test" can be removed. Storing V, W can take quite a lot
            # of space, especially when the input dimension d is high.
            # 'test':dcfssd_opt, 
            'test_result': dcksd_result, 'time_secs': t.secs}


def met_glksd_med_nzp1(P, Q, data_source, n, r, nz=1):
    """
    KSD-based model comparison test
    * Model P uses the approximate density with nz=1
    * Model Q uses the exact unnormalized marginal density
    """
    if not P.has_unnormalized_density() or not Q.has_unnormalized_density():
        # Not applicable. Return {}.
        return {}

    p = P.get_empirical_density(nz, seed=r+13)
    q = Q.get_unnormalized_density()
    # sample some data 
    datr = sample_pqr(None, None, data_source, n, r, only_from_r=True)

    # Start the timer here
    with util.ContextTimer() as t:
        medz = util.meddistance(datr.data(), subsample=1000)
        k = kernel.KGaussBumpL2(sigma2=medz**2, r=radius, scale=scale)

        dcksd= mct.DC_KSD(p, q, k, k, seed=r+11, alpha=alpha)
        dcksd_result = dcksd.perform_test(datr)

    return {
            # This key "test" can be removed. Storing V, W can take quite a lot
            # of space, especially when the input dimension d is high.
            # 'test':dcfssd_opt, 
            'test_result': dcksd_result, 'time_secs': t.secs}


def met_glksd_med_nzp100(P, Q, data_source, n, r):
    return met_glksd_med_nzp1(P, Q, data_source, n, r, nz=100)


def met_glksd_med_nzp1000(P, Q, data_source, n, r):
    return met_glksd_med_nzp1(P, Q, data_source, n, r, nz=1000)


def met_glksd_med_nzp100k(P, Q, data_source, n, r):
    return met_glksd_med_nzp1(P, Q, data_source, n, r, nz=(10)**5)

def met_glksd_med_nzpada(P, Q, data_source, n, r):
    return met_glksd_med_nzp1(P, Q, data_source, n, r, nz=30*n)

def met_glksd_med_nzp5n(P, Q, data_source, n, r):
    return met_glksd_med_nzp1(P, Q, data_source, n, r, nz=n*10)

def met_glksd_med_nzp1_cor(P, Q, data_source, n, r, nz=1):
    """
    KSD-based model comparison test with correction. 
    * Model P uses the approximate density with nz=1
    * Model Q uses the exact unnormalized marginal density
    """
    if not P.has_unnormalized_density() or not Q.has_unnormalized_density():
        # Not applicable. Return {}.
        return {}

    # p = P.get_empirical_density(nz, seed=r+13)
    # q = Q.get_unnormalized_density()
    # sample some data 
    datr = sample_pqr(None, None, data_source, n, r, only_from_r=True)

    # Start the timer here
    with util.ContextTimer() as t:
        medz = util.meddistance(datr.data(), subsample=1000)
        k = kernel.KGaussBumpL2(sigma2=medz**2, r=radius, scale=scale)

        ldcksd= mct.LDC_KSD(P, Q, k, k, approxq=False, nzp=nz, seed=r+11, alpha=alpha)
        ldcksd_result = ldcksd.perform_test(datr)

    return {
            # This key "test" can be removed. Storing V, W can take quite a lot
            # of space, especially when the input dimension d is high.
            # 'test':dcfssd_opt, 
            'test_result': ldcksd_result, 'time_secs': t.secs}


def met_glksd_med_nzp100_cor(P, Q, data_source, n, r):
    return met_glksd_med_nzp1_cor(P, Q, data_source, n, r, nz=100)


def met_glksd_med_nzp1000_cor(P, Q, data_source, n, r):
    return met_glksd_med_nzp1_cor(P, Q, data_source, n, r, nz=1000)


def met_glksd_med_nzpada_cor(P, Q, data_source, n, r):
    return met_glksd_med_nzp1_cor(P, Q, data_source, n, r, nz=2000)


def met_glksd_med_nzp5n_cor(P, Q, data_source, n, r):
    return met_glksd_med_nzp1_cor(P, Q, data_source, n, r, nz=n*10)


def met_glksd_med_nzp1_nzq1_cor(P, Q, data_source, n, r, nz=1):
    datr = sample_pqr(None, None, data_source, n, r, only_from_r=True)

    # Start the timer here
    with util.ContextTimer() as t:
        medz = util.meddistance(datr.data(), subsample=1000)
        k = kernel.KGaussBumpL2(sigma2=medz**2, r=radius, scale=scale)

        ldcksd= mct.LDC_KSD(P, Q, k, k, nzp=nz, nzq=nz, seed=r+11, alpha=alpha)
        ldcksd_result = ldcksd.perform_test(datr)

    return {
            # This key "test" can be removed. Storing V, W can take quite a lot
            # of space, especially when the input dimension d is high.
            # 'test':dcfssd_opt, 
            'test_result': ldcksd_result, 'time_secs': t.secs}


def met_glksd_med_nzp100_nzq100_cor(P, Q, data_source, n, r, nz=1):
    return met_glksd_med_nzp1_nzq1_cor(P, Q, data_source, n, r, nz=100)


def met_glksd_med_nzp1000_nzq1000_cor(P, Q, data_source, n, r, nz=1):
    return met_glksd_med_nzp1_nzq1_cor(P, Q, data_source, n, r, nz=1000)


def met_glksd_med_nzp2000_nzq2000_cor(P, Q, data_source, n, r, nz=1):
    return met_glksd_med_nzp1_nzq1_cor(P, Q, data_source, n, r, nz=2000)



# Define our custom Job, which inherits from base class IndependentJob
class Ex4Job(IndependentJob):
   
    def __init__(self, aggregator, P, Q, data_source, prob_label, rep, met_func, n):
        walltime = 60*59*24 
        #walltime = 60*59
        memory = int(n*1e-2) + 50

        IndependentJob.__init__(self, aggregator, walltime=walltime,
                               memory=memory)
        # P, P are kmod.model.Model
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

        logger.info("done. ex4: %s, prob=%s, r=%d, n=%d. Took: %.3g s "%(func_name,
            prob_label, r, n, t.secs))

        # save result
        fname = '%s-%s-n%d_r%d_a%.3f.p' \
                %(prob_label, func_name, n, r, alpha )
        glo.ex_save_result(ex, job_result, prob_label, fname)

# This import is needed so that pickle knows about the class Ex4Job.
# pickle is used when collecting the results from the submitted jobs.
from lkgof.ex.ex4_vary_n import Ex4Job
from lkgof.ex.ex4_vary_n import met_gmmd_med
from lkgof.ex.ex4_vary_n import met_gmmd_med_bounliphone
from lkgof.ex.ex4_vary_n import met_gksd_med
from lkgof.ex.ex4_vary_n import met_glksd_med_nzp100
from lkgof.ex.ex4_vary_n import met_glksd_med_nzp1000
from lkgof.ex.ex4_vary_n import met_glksd_med_nzp100k
from lkgof.ex.ex4_vary_n import met_glksd_med_nzpada
from lkgof.ex.ex4_vary_n import met_glksd_med_nzp1_cor
from lkgof.ex.ex4_vary_n import met_glksd_med_nzp100_cor
from lkgof.ex.ex4_vary_n import met_glksd_med_nzp1000_cor
from lkgof.ex.ex4_vary_n import met_glksd_med_nzpada_cor
from lkgof.ex.ex4_vary_n import met_glksd_med_nzp5n
from lkgof.ex.ex4_vary_n import met_glksd_med_nzp5n_cor
from lkgof.ex.ex4_vary_n import met_glksd_med_nzp1_nzq1_cor
from lkgof.ex.ex4_vary_n import met_glksd_med_nzp100_nzq100_cor
from lkgof.ex.ex4_vary_n import met_glksd_med_nzp1000_nzq1000_cor
from lkgof.ex.ex4_vary_n import met_glksd_med_nzp2000_nzq2000_cor

#--- experimental setting -----
ex = 4

# significance level of the test
alpha = 0.05

# repetitions for each sample size 
reps = 100

# kernel / model parameters
#radius = 15.
radius = 30. # dx=100
scale = 1


# tests to try
method_funcs = [ 
    #met_gfssdJ1_3sopt_tr20,
    #met_gfssdJ5_3sopt_tr20,
    #met_gmmd_med,
    met_gmmd_med_bounliphone,
    met_gksd_med,
    #met_glksd_med_nzp100,
    #met_glksd_med_nzp1000,
    #met_glksd_med_nzp5n,
    # met_glksd_med_nzpada,
    #met_glksd_med_nzp100_cor,
    met_glksd_med_nzp1000_cor,
    # met_glksd_med_nzpada_cor,
    #met_glksd_med_nzp5n_cor,
    # met_glksd_med_nzp100k,
    #met_glksd_med_nzp100_nzq100_cor,
    #met_glksd_med_nzp1000_nzq1000_cor,
    # met_glksd_med_nzp2000_nzq2000_cor,
   ]

# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting already exists.
is_rerun = False
#---------------------------


def make_ppca_prob(dx=10, dz=5, var=1.0,
                   ptbp=1., ptbq=1.,
                   seed=13, same=False):
    with util.NumpySeedContext(seed):
        weight = np.random.uniform(0, 1., [dx, dz])
    ppca_r = model.PPCASphereSupport(weight, var, radius)
    weightp = weight.copy()
    weightp[0, :] = weightp[0, :] + ptbp
    modelp = model.PPCASphereSupport(weightp, var, radius)
    if same:
        modelq = model.PPCASphereSupport(weightp, var,
                                      radius)
    else:
        weightq = weight.copy()
        weightq[0, :] = weightq[0, :] + ptbq
        modelq = model.PPCASphereSupport(weightq, var,
                                      radius)
    ds = ppca_r.get_datasource()

    return modelp, modelq, ds


def get_ns_pqrsource(prob_label):
    """
    Return (ns, P, Q, ds), a tuple of
    - ns: a list of sample sizes n's
    - P: a kmod.model.Model representing the model P
    - Q: a kmod.model.Model representing the model Q
    - ds: a DataSource. The DataSource generates sample from R.

    * (P, Q, ds) together specity a three-sample (or model comparison) problem.
    """

    prob2tuples = { 
        # A case where H0 (P is better) is true. 
        'ppca_h0_dx10_dz5_p1q2':
            # list of sample sizes
            ([100, 200, 300], ) + make_ppca_prob(dx=10, dz=5, ptbp=1., ptbq=2.,
                                                 same=False),
        'ppca_h0_dx10_dz5_p1q2_same':
            # list of sample sizes
            ([100, 200, 300], ) + make_ppca_prob(dx=10, dz=5, ptbp=1., ptbq=2.,
                                                 same=True),
        'ppca_h1_dx10_dz5_p2q1':
            # list of sample sizes
            ([100, 200, 300], ) + make_ppca_prob(dx=10, dz=5, ptbp=2., ptbq=1.,
                                                 same=False),
        'ppca_h0_dx50_dz10_p1q2':
        # list of sample sizes
            ([i*100 for i in range(1, 3+1)], ) + make_ppca_prob(dx=50, dz=10,
                                                                ptbp=1., ptbq=2.,
                                                                same=False),
        'ppca_h1_dx50_dz10_p2q1':
            # list of sample sizes
            ([i*100 for i in range(1, 3+1)], ) + make_ppca_prob(dx=50, dz=10, ptbp=2.,
                                                                ptbq=1.),
        'ppca_h1_dx50_dz10_p2q5e-1':
            # list of sample sizes
            ([i*100 for i in range(1, 3+1)], ) + make_ppca_prob(dx=50, dz=10, ptbp=2.,
                                                                ptbq=0.5),
        'ppca_h1_dx100_dz10_p2q1':
            # list of sample sizes
            ([i*100 for i in range(1, 3+1)], ) + make_ppca_prob(dx=100, dz=10, ptbp=2.,
                                                                ptbq=1.),
        'ppca_h1_dx100_dz10_p1q5e-1':
            # list of sample sizes
            ([i*100 for i in range(1, 3+1)], ) + make_ppca_prob(dx=100, dz=10, ptbp=1.,
                                                                ptbq=0.5),
        'ppca_h0_dx100_dz10_p5e-1q1':
            # list of sample sizes
            ([i*100 for i in range(1, 3+1)], ) + make_ppca_prob(dx=100, dz=10, ptbp=0.5,
                                                                ptbq=1),
        'ppca_h0_dx100_dz10_p1_same':
            # list of sample sizes
            ([i*100 for i in range(1, 3+1)], ) + make_ppca_prob(dx=100, dz=10, ptbp=1,
                                                                ptbq=1, same=True),
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
                    job = Ex4Job(SingleResultAggregator(), P, Q, ds, prob_label,
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

