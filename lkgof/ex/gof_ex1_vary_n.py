"""Simulation to test the test power vs increasing sample size"""

__author__ = 'wittawat'

import lkgof
import lkgof.data as data
from lkgof import glo
from lkgof import density
from lkgof import goftest as gof
from lkgof import model
import kgof.intertst as tgof
import kgof.mmd as mgof
import kgof.util as util 
import lkgof.kernel as kernel 

# need independent_jobs package 
# https://github.com/karlnapf/independent-jobs
# The independent_jobs and kgof have to be in the global search path (.bashrc)
import independent_jobs as inj
from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.SingleResult import SingleResult
from independent_jobs.aggregators.SingleResultAggregator import SingleResultAggregator
from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.tools.Log import logger
import logging
import math
#import numpy as np
import autograd.numpy as np
import os
import sys 
import time

"""
All the job functions return a dictionary with the following keys:
    - goftest: test object. (may or may not return)
    - test_result: the result from calling perform_test(te).
    - time_secs: run time in seconds 
"""


def job_kstein_med(p, data_source, tr, te, r):
    """
    Kernel Stein discrepancy test of Liu et al., 2016 and Chwialkowski et al.,
    2016. Use full sample. Use Gaussian kernel.
    """
    # full data
    data = tr + te
    X = data.data()
    if not p.has_unnormalized_density():
        msg = "the model must have an unnormalized marignal"
        raise AttributeError(msg)
    um = p.get_unnormalized_density()
    var_type_disc = um.var_type_disc
    with util.ContextTimer() as t:
        if var_type_disc:
            k = kernel.KHamming(um.n_values)
        else:
            # median heuristic 
            med = util.meddistance(X, subsample=1000)
            k = kernel.KGauss(med**2)
        kstein = gof.KernelSteinTest(um, k, alpha=alpha, n_simulate=1000, seed=r)
        kstein_result = kstein.perform_test(data)
    return {'test_result': kstein_result, 'time_secs': t.secs}


def job_mmd_med(p, data_source, tr, te, r, ny=None):
    """
    MMD test of Gretton et al., 2012 used as a goodness-of-fit test.
    Require the ability to sample from p i.e., the UnnormalizedDensity p has 
    to be able to return a non-None from get_datasource()
    """
    # full data
    data = tr + te
    X = data.data()
    with util.ContextTimer() as t:
        # median heuristic 
        pds = p.get_datasource()
        if ny is None:
            datY = pds.sample(data.sample_size(), seed=r+294)
        else:
            datY = pds.sample(ny, seed=r+294)
        Y = datY.data()
        emp_p = p.get_empirical_density(1)
        if emp_p.var_type_disc:
            k = kernel.KHamming(emp_p.n_values)
        else:
            XY = np.vstack((X, Y))
            # If p, q differ very little, the median may be very small, rejecting H0
            # when it should not?
            medx = util.meddistance(X, subsample=1000)
            medy = util.meddistance(Y, subsample=1000)
            medxy = util.meddistance(XY, subsample=1000)
            med_avg = (medx+medy+medxy)/3.0
            k = kernel.KGauss(med_avg**2)

        mmd_test = mgof.QuadMMDGof(p, k, n_permute=400, alpha=alpha, seed=r)
        mmd_result = mmd_test.perform_test(data)
    return {'test_result': mmd_result, 'time_secs': t.secs}


def job_lksd_med(p, data_source, tr, te, r, nz=1, null_sim=None):
    """
    A latent variable model version of Kernel Stein discrepancy test
    of Liu et al., 2016 and Chwialkowski et al.,
    2016. Use full sample. Use Gaussian kernel.
    """
    # full data
    data = tr + te
    X = data.data()
    with util.ContextTimer() as t:
        # median heuristic
        med = util.meddistance(X, subsample=1000)
        k = kernel.KGauss(med**2)
        kstein = gof.LatentKernelSteinTest(p, k, alpha=alpha,
                                           n_simulate=400, seed=r)
        kstein_result = kstein.perform_test(data)
    return {'test_result': kstein_result, 'time_secs': t.secs}


# Define our custom Job, which inherits from base class IndependentJob
class Ex1Job(IndependentJob):

    def __init__(self, aggregator, p, data_source, prob_label, rep, job_func, n):
        # walltime = 60*59*24 
        walltime = 60*59
        memory = 54272

        IndependentJob.__init__(self, aggregator, walltime=walltime,
                                memory=memory)
        # p: an UnnormalizedDensity
        self.p = p
        self.data_source = data_source
        self.prob_label = prob_label
        self.rep = rep
        self.job_func = job_func
        self.n = n

    # we need to define the abstract compute method. It has to return an instance
    # of JobResult base class
    def compute(self):

        p = self.p
        data_source = self.data_source 
        r = self.rep
        n = self.n
        job_func = self.job_func
        data = data_source.sample(n, seed=r)
        with util.ContextTimer() as t:
            tr, te = data.split_tr_te(tr_proportion=tr_proportion, seed=r+21 )
            prob_label = self.prob_label
            logger.info("computing. %s. prob=%s, r=%d,\
                    n=%d"%(job_func.__name__, prob_label, r, n))

            job_result = job_func(p, data_source, tr, te, r)

            # create ScalarResult instance
            result = SingleResult(job_result)
            # submit the result to my own aggregator
            self.aggregator.submit_result(result)
            func_name = job_func.__name__
        logger.info("done. ex2: %s, prob=%s, r=%d, n=%d. Took: %.3g s "%(func_name,
            prob_label, r, n, t.secs))

        # save result
        fname = '%s-%s-n%d_r%d_a%.3f_trp%.2f.p' \
                %(prob_label, func_name, n, r, alpha, tr_proportion)
        glo.ex_save_result(ex, job_result, prob_label, fname)


# This import is needed so that pickle knows about the class Ex1Job.
# pickle is used when collecting the results from the submitted jobs.
from lkgof.ex.ex1_vary_n import Ex1Job
from lkgof.ex.ex1_vary_n import job_lksd_med
from lkgof.ex.ex1_vary_n import job_kstein_med
from lkgof.ex.ex1_vary_n import job_mmd_med


#--- experimental setting -----
ex = 1

# significance level of the test
alpha = 0.05

# Proportion of training sample relative to the full sample size n
tr_proportion = 0.2

# repetitions for each sample size 
reps = 300

# tests to try
method_job_funcs = [ 
    job_lksd_med,
    job_kstein_med,
    job_mmd_med,
]

# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting of (ni, r) already exists.
is_rerun = False
#---------------------------


def get_mbb_model(dx=10, seed=13):
    with util.NumpySeedContext(seed):
        alphas = np.random.uniform(1, 2, [dx])
        betas = np.random.uniform(1, 3, [dx])

    model = model.MultivariateBernBeta(alphas, betas)
    alphas_ = alphas
    alphas_[0] += 2.
    ds = model.MultivariateBernBeta(alphas_, betas).get_datasource()
    #probs = model.get_unnormalized_density().probs
    #with util.NumpySeedContext(seed=seed+2):
    #    probs[0] = np.random.uniform(0, 1, [1])
    # ds = data.DSMultivariateBern(probs)
    return model, ds


def get_ppca_model(dx=10, dz=1, var=1.0, ptb=1., seed=13):
    """
    Get a PPCA model
    """
    with util.NumpySeedContext(seed):
        weight = np.random.random([dx, dz])
    weight_ = weight.copy()
    weight_[0, :] = weight_[0, :] + ptb
    ppca = model.PPCA(weight_, var)
    ds = model.PPCA(weight, var).get_datasource()
    return ppca, ds


def get_ns_pqsource(prob_label):
    """
    Return (ns, p, ds), a tuple of
    where
    - ns: a list of sample sizes
    - p: a LatentVariableModel represnting a latent model p
    - ds: a DataSource, each corresponding to one parameter setting.
        The DataSource generates sample from q.
    """

    prob2tuples = {
        'ppca_dx10_dz5_h0':
            ([i*100 for i in range(1,5+1)],) +
             get_ppca_model(dx=10, dz=5, ptb=0.),
        'ppca_dx10_dz5_mp1':
            ([i*100 for i in range(1,3+1)],) +
             get_ppca_model(dx=10, dz=5, ptb=1.),
        'ppca_dx30_dz10_h0':
            ([i*100 for i in range(1,3+1)],) +
             get_ppca_model(dx=30, dz=10, ptb=0.),
        'ppca_dx30_dz10_h1':
            ([i*100 for i in range(1,3+1)],) +
             get_ppca_model(dx=30, dz=10, ptb=1.),
    }
    if prob_label not in prob2tuples:
        raise ValueError('Unknown problem label. Need to be one of %s'%str(prob2tuples.keys()) )
    return prob2tuples[prob_label]


def run_problem(prob_label):
    """Run the experiment"""
    ns, p, ds = get_ns_pqsource(prob_label)
    # ///////  submit jobs //////////
    # create folder name string
    #result_folder = glo.result_folder()
    from lkgof.config import expr_configs
    tmp_dir = expr_configs['scratch_path']
    foldername = os.path.join(tmp_dir, 'lkgof_slurm', 'e%d'%ex)
    logger.info("Setting engine folder to %s" % foldername)

    # create parameter instance that is needed for any batch computation engine
    logger.info("Creating batch parameter instance")
    batch_parameters = BatchClusterParameters(
        foldername=foldername, job_name_base="e%d_"%ex, parameter_prefix="")

    # Use the following line if Slurm queue is not used.
    #engine = SerialComputationEngine()
    #engine = SlurmComputationEngine(batch_parameters, partition='wrkstn,compute')
    partitions = expr_configs['slurm_partitions']
    if partitions is None:
        engine = SlurmComputationEngine(batch_parameters)
    else:
        engine = SlurmComputationEngine(batch_parameters, partition=partitions)
    # engine = SlurmComputationEngine(batch_parameters)
    n_methods = len(method_job_funcs)
    # repetitions x len(ns) x #methods
    aggregators = np.empty((reps, len(ns), n_methods ), dtype=object)
    for r in range(reps):
        for ni, n in enumerate(ns):
            for mi, f in enumerate(method_job_funcs):
                # name used to save the result
                func_name = f.__name__
                fname = '%s-%s-n%d_r%d_a%.3f_trp%.2f.p' \
                        %(prob_label, func_name, n, r, alpha, tr_proportion)
                if not is_rerun and glo.ex_file_exists(ex, prob_label, fname):
                    logger.info('%s exists. Load and return.'%fname)
                    job_result = glo.ex_load_result(ex, prob_label, fname)

                    sra = SingleResultAggregator()
                    sra.submit_result(SingleResult(job_result))
                    aggregators[r, ni, mi] = sra
                else:
                    # result not exists or rerun

                    # p: an UnnormalizedDensity object
                    job = Ex1Job(SingleResultAggregator(), p, ds, prob_label,
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
            for mi, f in enumerate(method_job_funcs):
                logger.info("Collecting result (%s, r=%d, n=%rd)" %
                        (f.__name__, r, n))
                # let the aggregator finalize things
                aggregators[r, ni, mi].finalize()

                # aggregators[i].get_final_result() returns a SingleResult instance,
                # which we need to extract the actual result
                job_result = aggregators[r, ni, mi].get_final_result().result
                job_results[r, ni, mi] = job_result

    #func_names = [f.__name__ for f in method_job_funcs]
    #func2labels = exglobal.get_func2label_map()
    #method_labels = [func2labels[f] for f in func_names if f in func2labels]

    # save results 
    results = {'job_results': job_results, 'data_source': ds, 
            'alpha': alpha, 'repeats': reps, 'ns': ns,
            'p': p,
            'tr_proportion': tr_proportion,
            'method_job_funcs': method_job_funcs, 'prob_label': prob_label,
            }
    
    # class name 
    fname = 'ex%d-%s-me%d_rs%d_nmi%d_nma%d_a%.3f_trp%.2f.p' \
        %(ex, prob_label, n_methods, reps, min(ns), max(ns), alpha,
                tr_proportion)

    glo.ex_save_result(ex, results, fname)
    logger.info('Saved aggregated results to %s'%fname)


def main():
    if len(sys.argv) != 2:
        print('Usage: %s problem_label'%sys.argv[0])
        sys.exit(1)
    prob_label = sys.argv[1]
    run_problem(prob_label)

if __name__ == '__main__':
    main()
