"""
Module for testing density module.
"""

__author__ = 'wittawat'

import numpy as np
import lkgof.density as density
import scipy.stats as stats
import lkgof.util as util

import unittest


class TestBinomial(unittest.TestCase):
    def setUp(self):
        pass


    def test_eval_den(self):
        n = 13
        max_n = 10
        d = 4
        for s in [12, 38, 982, 3]:
            with util.NumpySeedContext(seed=s):
                n_values = np.random.randint(1, max_n, d)
                probs = np.random.rand(d)

                X = np.random.randint(0, max_n, (n, d))
                probs1 = [stats.binom.pmf(X[:, j], n=n_values[j], p=probs[j])
                         for j in range(d)]
                P1 = np.prod(np.array(probs1), axis=0)

                dist = density.Binomial(n_trials=n_values, probs=probs)
                P2 = dist.eval_den(X)

                # check correctness 
                np.testing.assert_almost_equal(P1, P2)

    def test_log_normalized_den(self):
        n = 23
        max_n = 17
        d = 10
        for s in [13, 38, 92, 3]:
            with util.NumpySeedContext(seed=s):
                n_values = np.random.randint(1, max_n, d)
                probs = np.random.rand(d)

                X = np.random.randint(0, max_n, (n, d))
                probs1 = [stats.binom.logpmf(X[:, j], n=n_values[j], p=probs[j])
                         for j in range(d)]
                P1 = np.sum(np.array(probs1), axis=0)

                dist = density.Binomial(n_trials=n_values, probs=probs)
                P2 = dist.log_normalized_den(X)

                # check correctness 
                np.testing.assert_almost_equal(P1, P2)




    def tearDown(self):
        pass


if __name__ == '__main__':
   unittest.main()

